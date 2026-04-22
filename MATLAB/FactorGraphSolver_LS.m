%% ========================================================================
%  DEFINICE TŘÍDY ŘEŠIČE (OPTIMALIZOVANÁ VERZE)
% =========================================================================
classdef FactorGraphSolver_LS < handle
    properties
        % Model
        model
        nx
        nz

        % Data
        meas
        N

        % Parametry (předpočítané inverze pro rychlost)
        S_Q_inv
        S_R_inv
        invP0chol

        % Stavové proměnné
        x0 % Prior mean (počátek batche)
        P0 % Prior covariance
        states % Odhadnutá trajektorie v okně

        % Cache pro zrychlení
        last_Lambda % Uložená informační matice (Hessian)
    end

    methods
        function obj = FactorGraphSolver_LS(model, meas, R, Q, x0, P0)
            % KONSTRUKTOR - volá se pouze JEDNOU před cyklem
            obj.model = model;
            obj.nx = model.nx;
            obj.nz = model.nz;

            obj.meas = meas;
            obj.N = size(meas, 2);
            obj.x0 = x0;
            obj.P0 = P0;

            % Předpočítání inverzních Cholesky faktorů (drahá operace)
            % S_Q_inv * S_Q_inv' = inv(Q)
            obj.S_Q_inv = inv(chol(Q, 'lower'));
            obj.S_R_inv = inv(chol(R, 'lower'));

            % Inicializace prioru
            obj.invP0chol = inv(chol(P0, 'lower'));

            % Inicializace stavů (Warm start - prostá predikce modelem)
            obj.states = zeros(obj.N, obj.nx);
            obj.states(1,:) = x0';
            obj.create_init_state();
        end

        function update_problem(obj, meas_new, x0_new, P0_new)
            % === METODA PRO RECYKLACI SOLVERU ===
            % Místo 'new FactorGraphSolver' jen vyměníme data.
            % Inverze Q a R zůstávají v paměti.

            obj.meas = meas_new;
            obj.N = size(meas_new, 2); % Kdyby se měnila velikost batche
            obj.x0 = x0_new;
            obj.P0 = P0_new;

            % Prior kovariance se mění každou iteraci -> musíme přepočítat
            % Pro malé matice (4x4) je to zanedbatelné.
            obj.invP0chol = inv(chol(P0_new, 'lower'));

            % Reset stavů (Warm start z nového prioru)
            if size(obj.states, 1) ~= obj.N
                obj.states = zeros(obj.N, obj.nx);
            end
            obj.states(1,:) = x0_new';
            obj.create_init_state();

            % Vymazání cache, protože se změnila data
            obj.last_Lambda = [];
        end

        function create_init_state(obj)
            % Jednoduchá dopředná simulace pro inicializaci
            for i = 2:obj.N
                obj.states(i,:) = obj.model.F(obj.states(i-1,:)')';
            end
        end

        % --- Pomocné indexování ---
        function idx = state_idx(obj, i), idx = obj.nx * (i-1) + 1; end
        function idx = dyn_idx(obj, i), idx = obj.nx * (i - 2) + 1; end
        function idx = meas_idx(obj, i), idx = obj.nx * (obj.N - 1) + obj.nz * (i - 1) + 1; end

        function L = create_L(obj)
            % Sestavení velké řídké Jacobiho matice L
            % Optimalizováno pomocí tripletů (rows, cols, vals)

            num_dyn = obj.N - 1;
            num_meas = obj.N;

            % Odhad počtu nenulových prvků (prealokace pro rychlost)
            nnz_est = num_dyn*(2*obj.nx^2) + num_meas*(obj.nz*obj.nx) + obj.nx^2;

            rows = zeros(nnz_est, 1);
            cols = zeros(nnz_est, 1);
            vals = zeros(nnz_est, 1);
            k = 1;

            % --- 1. Dynamika (Motion Model) ---
            for i = 2:obj.N
                % Derivace podle x_{i-1} -> -S_Q_inv * F_j
                F_j = obj.model.dFdX(obj.states(i-1,:)');
                block = -obj.S_Q_inv * F_j;

                [r,c,v] = find(block);
                num = length(r);
                rows(k:k+num-1) = obj.dyn_idx(i) + r - 1;
                cols(k:k+num-1) = obj.state_idx(i-1) + c - 1;
                vals(k:k+num-1) = v;
                k = k + num;

                % Derivace podle x_i -> S_Q_inv
                block = obj.S_Q_inv;
                [r,c,v] = find(block);
                num = length(r);
                rows(k:k+num-1) = obj.dyn_idx(i) + r - 1;
                cols(k:k+num-1) = obj.state_idx(i) + c - 1;
                vals(k:k+num-1) = v;
                k = k + num;
            end

            % --- 2. Měření (Measurement Model) ---
            for i = 1:obj.N
                % Derivace podle x_i -> -S_R_inv * H_j
                H_j = obj.model.dHdX(obj.states(i,:)');
                block = -obj.S_R_inv * H_j;

                [r,c,v] = find(block);
                num = length(r);
                rows(k:k+num-1) = obj.meas_idx(i) + r - 1;
                cols(k:k+num-1) = obj.state_idx(i) + c - 1;
                vals(k:k+num-1) = v;
                k = k + num;
            end

            % --- 3. Prior (Marginalizace) ---
            prior_start = obj.nx*(obj.N-1) + obj.nz*obj.N + 1;
            block = obj.invP0chol;
            [r,c,v] = find(block);
            num = length(r);
            rows(k:k+num-1) = prior_start + r - 1;
            cols(k:k+num-1) = obj.state_idx(1) + c - 1;
            vals(k:k+num-1) = v;
            k = k + num;

            % Vytvoření sparse matice
            L = sparse(rows(1:k-1), cols(1:k-1), vals(1:k-1), ...
                prior_start + obj.nx - 1, obj.nx * obj.N);
        end

        function y = create_y(obj, state_vec_col)
            if nargin < 2 || isempty(state_vec_col)
                state_data = obj.states;
            else
                state_data = reshape(state_vec_col, [obj.nx, obj.N])';
            end

            num_rows = obj.nx*(obj.N-1) + obj.nz*obj.N + obj.nx;
            y = zeros(num_rows, 1);

            % Dynamika
            for i = 2:obj.N
                e = state_data(i,:)' - obj.model.F(state_data(i-1,:)');
                y(obj.dyn_idx(i) : obj.dyn_idx(i) + obj.nx - 1) = obj.S_Q_inv * e;
            end

            % Měření
            for i = 1:obj.N
                e = obj.meas(:,i) - obj.model.H(state_data(i,:)');
                y(obj.meas_idx(i) : obj.meas_idx(i) + obj.nz - 1) = obj.S_R_inv * e;
            end

            % Prior
            e_prior = state_data(1,:)' - obj.x0;
            y(end-obj.nx+1 : end) = obj.invP0chol * e_prior;
        end


        function opt(obj)
            % =============================================================
            % OPTIMALIZAČNÍ ALGORITMUS - GAUSS NEWTON (GTSAM STYLE)
            % =============================================================
            % Parametry odpovídající gtsam::GaussNewtonParams
            maxIter = 100;
            relativeErrorTol = 1e-5; % Zastavit, pokud se chyba změní o méně než toto
            absoluteErrorTol = 1e-5; % Zastavit, pokud je velikost kroku malá

            % V GTSAM se chyba počítá jako 0.5 * sum(residuals^2)
            y_resid = obj.create_y();
            current_error = 0.5 * (y_resid' * y_resid);

            disp(['Initial Error: ', num2str(current_error)]);

            for iter = 1:maxIter
                % 1. Linearizace (Jacobian a residua)
                L = obj.create_L();
                y_resid = obj.create_y();

                % 2. Sestavení Normálních rovnic: (J'J) * d = -J'e
                % GTSAM používá Choleského rozklad nebo QR, v Matlabu operátor '\'
                % vybere nejvhodnější metodu.

                H = L' * L;       % Hessian (J'J)
                g = -(L' * y_resid); % Gradient (-J'e)

                try
                    d = H \ g;
                catch
                    warning('Matice H je singulární. Gauss-Newton selhal.');
                    break;
                end

                current_states_vec = reshape(obj.states', [], 1);
                new_states_vec = current_states_vec + d;

                % Aktualizace stavu
                obj.states = reshape(new_states_vec, [obj.nx, obj.N])';

                % 3. Výpočet nové chyby pro kontrolu konvergence
                y_new = obj.create_y();
                new_error = 0.5 * (y_new' * y_new);

                disp(['Iter ', num2str(iter), ': Error ', num2str(new_error)]);

                % 4. Podmínky ukončení (Termination criteria)

                % A) Relativní pokles chyby
                % Všimněte si 'abs', GN může někdy chybu i zvýšit, pokud je krok špatný
                if abs(current_error - new_error) < relativeErrorTol
                    disp('Konvergováno (Relative Error).');
                    break;
                end

                % B) Absolutní velikost kroku (dx)
                if norm(d) < absoluteErrorTol
                    disp('Konvergováno (Absolute Step).');
                    break;
                end

                % Aktualizace chyby pro další krok
                current_error = new_error;
            end

            % CACHING PRO KOVARIANCE
            % (Stejné jako předtím - přepočítat L v optimu)
            L_final = obj.create_L();
            obj.last_Lambda = L_final' * L_final;
        end

        function P = compute_covariance(obj)
            % Využijeme cacheovanou matici Lambda, pokud existuje
            if isempty(obj.last_Lambda)
                L = obj.create_L();
                Lambda = L' * L;
            else
                Lambda = obj.last_Lambda;
            end

            % Výpočet kovariance P = inv(Lambda)
            % Pro batch size ~50-100 je matice Lambda malá (200x200 až 400x400).
            % Konverze na 'full' a použití dense inverze je v MATLABu často
            % rychlejší než sparse inverze pro tuto velikost.

            Lambda_full = full(Lambda);

            % Malá regularizace pro numerickou stabilitu
            regularization = 1e-9 * eye(size(Lambda_full));
            P = inv(Lambda_full + regularization);
        end
    end
end