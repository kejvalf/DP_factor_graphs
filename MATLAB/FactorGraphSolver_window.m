classdef FactorGraphSolver_window < handle
    properties
        model
        nx
        nz
        meas
        N
        
        % Předpočítané matice
        S_Q_inv
        S_R_inv
        invP0chol
        
        x0 % Prior state (mean)
        P0 % Prior covariance
        
        states % Aktuální odhad stavů
        
        % Cache pro zrychlení výpočtu kovariance
        last_Lambda 
    end

    methods
        function obj = FactorGraphSolver_window(model, meas, R, Q, x0, P0)
            % Konstruktor - volá se pouze JEDNOU na začátku
            obj.model = model;
            obj.nx = model.nx;
            obj.nz = model.nz;
            obj.meas = meas;
            obj.N = size(meas, 2);
            obj.x0 = x0;
            obj.P0 = P0;

            % Předpočítání inverzních faktorů šumu (drahá operace, děláme 1x)
            obj.S_Q_inv = inv(chol(Q, 'lower'));
            obj.S_R_inv = inv(chol(R, 'lower'));
            
            % Inicializace prioru
            obj.invP0chol = inv(chol(P0, 'lower'));

            % Inicializace stavů
            obj.states = zeros(obj.N, obj.nx);
            obj.states(1,:) = x0';
            obj.create_init_state();
        end

        function update_problem(obj, meas_new, x0_new, P0_new, initial_guess)
            % === RYCHLÝ UPDATE PRO POSUVNÉ OKNO ===
            % Místo vytváření nového objektu jen přepíšeme data
            obj.meas = meas_new;
            obj.x0 = x0_new;
            
            % Přepočet prior kovariance (toto se mění v každém okně)
            % Pro malé matice (4x4) je to rychlé
            obj.invP0chol = inv(chol(P0_new, 'lower'));
            
            % Warm Start: Použijeme předchozí trajektorii jako odhad
            if nargin > 4 && ~isempty(initial_guess)
                obj.states = initial_guess; % initial_guess musí být [N x nx]
            else
                obj.states(1,:) = x0_new';
                obj.create_init_state();
            end
            
            % Vymazání cache, protože se změnila data
            obj.last_Lambda = [];
        end

        function create_init_state(obj)
            % Pokud nemáme warm start, dopočítáme z modelu
            for i = 2:obj.N
                obj.states(i,:) = obj.model.F(obj.states(i-1,:)')';

                % Použití naměřené rychlosti pro počáteční odhad odhad
                % x_prev = obj.states(i-1,:)';
                % % 1. Vnutíme naměřenou rychlost z minulého kroku
                % % (meas(2, :) je z_vx, meas(3, :) je z_vy)
                % x_prev(3:4) = obj.meas(2:3, i-1);
                % % 2. Provedeme krok dynamikou (posun polohy pomocí správné rychlosti)
                % x_new = obj.model.F(x_prev);
                % % 3. Do nového stavu opět rovnou vnutíme aktuální naměřenou rychlost
                % x_new(3:4) = obj.meas(2:3, i);
                % % Uložíme jako výchozí odhad pro řešič
                % obj.states(i,:) = x_new';
            end
        end

        % Pomocné indexovací funkce
        function idx = state_idx(obj, i)
            idx = obj.nx * (i-1) + 1;
        end
        function idx = dyn_idx(obj, i)
            idx = obj.nx * (i - 2) + 1;
        end
        function idx = meas_idx(obj, i)
            dynamics_rows = obj.nx * (obj.N - 1);
            idx = dynamics_rows + obj.nz * (i - 1) + 1;
        end

        function L = create_L(obj)
            % Sestavení velké řídké Jacobiho matice L
            % Používáme triplet formát (row, col, val) pro rychlost
            
            num_dyn = obj.N - 1;
            num_meas = obj.N;
            
            % Odhad počtu nenulových prvků (prealokace)
            nz_est = num_dyn*(2*obj.nx^2) + num_meas*(obj.nz*obj.nx) + obj.nx^2;
            
            rows = zeros(nz_est, 1);
            cols = zeros(nz_est, 1);
            vals = zeros(nz_est, 1);
            k = 1;

            % --- Dynamika ---
            for i = 2:obj.N
                % Derivace podle x_{i-1} -> -S_Q_inv * F_j
                F_j = obj.model.dFdX(obj.states(i-1,:)');
                block = -obj.S_Q_inv * F_j;
                
                r_idx = obj.dyn_idx(i);
                c_idx = obj.state_idx(i-1);
                
                [r, c, v] = find(block); % Najde nenulové prvky v malém bloku
                n = length(r);
                rows(k:k+n-1) = r_idx + r - 1;
                cols(k:k+n-1) = c_idx + c - 1;
                vals(k:k+n-1) = v;
                k = k + n;

                % Derivace podle x_i -> S_Q_inv
                block = obj.S_Q_inv;
                r_idx = obj.dyn_idx(i);
                c_idx = obj.state_idx(i);
                
                [r, c, v] = find(block);
                n = length(r);
                rows(k:k+n-1) = r_idx + r - 1;
                cols(k:k+n-1) = c_idx + c - 1;
                vals(k:k+n-1) = v;
                k = k + n;
            end

            % --- Měření ---
            for i = 1:obj.N
                % Derivace podle x_i -> -S_R_inv * H_j
                H_j = obj.model.dHdX(obj.states(i,:)');
                block = -obj.S_R_inv * H_j;
                
                r_idx = obj.meas_idx(i);
                c_idx = obj.state_idx(i);
                
                [r, c, v] = find(block);
                n = length(r);
                rows(k:k+n-1) = r_idx + r - 1;
                cols(k:k+n-1) = c_idx + c - 1;
                vals(k:k+n-1) = v;
                k = k + n;
            end

            % --- Prior ---
            prior_row_start = obj.nx*(obj.N-1) + obj.nz*obj.N + 1;
            block = obj.invP0chol;
            c_idx = obj.state_idx(1);
            
            [r, c, v] = find(block);
            n = length(r);
            rows(k:k+n-1) = prior_row_start + r - 1;
            cols(k:k+n-1) = c_idx + c - 1;
            vals(k:k+n-1) = v;
            k = k + n;

            % Oříznutí přebytečných nul a vytvoření sparse matice
            L = sparse(rows(1:k-1), cols(1:k-1), vals(1:k-1), ...
                       prior_row_start + obj.nx - 1, obj.nx * obj.N);
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
            % OPTIMALIZAČNÍ ALGORITMUS: Adaptivní Levenberg-Marquardt
            % (Přesná obdoba GTSAM LevenbergMarquardtOptimizer)
            % =============================================================
            maxIter = 100;   % Zvýšeno, z dálky to potřebuje více iterací
            tol     = 1e-6; 
            
            % Parametry odpovídající GTSAM nastavení
            lambda = 1e-4;       % params.lambdaInitial
            lambdaFactor = 2.0;  % params.lambdaFactor
            
            % Výpočet počáteční chyby
            y_resid = obj.create_y();
            cost = y_resid' * y_resid;

            for iter = 1:maxIter
                % Sestavení systému L a M = L'*L
                L = obj.create_L();
                M = L' * L;
                rhs = -(L' * y_resid);
                
                % Identita pro Levenberg-Marquardt tlumení
                % (Používáme speye, abychom zamezili singulární matici, i když jsou gradienty 0)
                I = speye(size(M));
                
                step_accepted = false;
                
                % Vnitřní smyčka: hledáme správné lambda
                while ~step_accepted && lambda < 1e6
                    % Přičtení lambda na diagonálu
                    M_damped = M + lambda * I;
                    
                    % Výpočet kroku
                    d = M_damped \ rhs;
                    
                    if norm(d) < tol
                        step_accepted = true;
                        break;
                    end
                    
                    % Testovací krok
                    current_states_vec = reshape(obj.states',[], 1);
                    new_states_vec = current_states_vec + d;
                    
                    % Vypočteme novou chybu
                    y_new = obj.create_y(new_states_vec);
                    cost_new = y_new' * y_new;

                    % Podmínka přijetí kroku
                    if cost_new < cost
                        % KROK PŘIJAT (Chyba se zmenšila)
                        obj.states = reshape(new_states_vec,[obj.nx, obj.N])';
                        y_resid = y_new;
                        cost = cost_new;
                        
                        % Snížíme lambda (v GTSAM se dělí)
                        lambda = max(1e-7, lambda / lambdaFactor); 
                        step_accepted = true;
                    else
                        % KROK ZAMÍTNUT (Chyba vzrostla)
                        % Zvětšíme lambda a zkusíme to v cyklu znovu 
                        % (tohle dělá optimalizátor robustním!)
                        lambda = lambda * lambdaFactor;
                    end
                end

                % Kontrola konvergence
                if ~step_accepted || norm(d) < tol
                    break;
                end
            end
            
            %fprintf('Optimalizace dokončena v %d. iteraci.\n', iter);
            
            % CACHING (důležité pro compute_covariance)
            L_final = obj.create_L();
            obj.last_Lambda = L_final' * L_final;
        end

        function P = compute_covariance(obj)
            % Využijeme cacheovanou matici Lambda z metody opt()
            if isempty(obj.last_Lambda)
                L = obj.create_L();
                Lambda = L' * L;
            else
                Lambda = obj.last_Lambda;
            end
            
            % Pro malá okna (N < 50) je konverze na full a inv() často 
            % rychlejší než sparse inv, protože Matlab má optimalizované dense operace.
            if obj.N <= 50
                P = inv(full(Lambda) + 1e-9*eye(size(Lambda)));
            else
                P = inv(Lambda + 1e-9*speye(size(Lambda)));
            end
        end
    end
end