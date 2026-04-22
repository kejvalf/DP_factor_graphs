classdef TrajectoryFilters < handle
    % TRAJECTORYFILTER Univerzální třída pro KF, EKF, IEKF, UKF a RTS.
    
    properties
        Model   % Struktura: .F(x), .H(x), .dFdX(x), .dHdX(x)
        Q       % Matice šumu procesu
        R       % Matice šumu měření
        x0      % Počáteční stav
        P0      % Počáteční kovariance
    end
    
    methods
        function obj = TrajectoryFilters(model, Q, R, x0, P0)
            obj.Model = model;
            obj.Q = Q;
            obj.R = R;
            obj.x0 = x0;
            obj.P0 = P0;
        end
        
        %% 1. LINEÁRNÍ KALMANŮV FILTR (KF)
        function [x_est, P_hist] = runKF(obj, measurements, F_override, H_override)
            N = size(measurements, 2);
            nx = length(obj.x0);
            
            if nargin > 2 && ~isempty(F_override), F = F_override;
            elseif isfield(obj.Model, 'dFdX'), F = obj.Model.dFdX(obj.x0);
            else, F = obj.Model.F; end
            
            if nargin > 3 && ~isempty(H_override), H = H_override;
            elseif isfield(obj.Model, 'dHdX'), H = obj.Model.dHdX(obj.x0);
            else, H = obj.Model.H; end
            
            x_est = zeros(nx, N); x_est(:, 1) = obj.x0;
            P = obj.P0; P_hist = zeros(nx, nx, N); P_hist(:, :, 1) = P;
            I = eye(nx);
            
            for k = 2:N
                x_pred = F * x_est(:, k-1);
                P_pred = F * P * F' + obj.Q;
                z_meas = measurements(:, k);
                y_res = z_meas - H * x_pred;
                S = H * P_pred * H' + obj.R;
                K = P_pred * H' / S;
                x_est(:, k) = x_pred + K * y_res;
                P = (I - K * H) * P_pred;
                P_hist(:, :, k) = P;
            end
        end
        
        %% 2. EKF (Extended Kalman Filter)
        function [x_est, P_hist] = runEKF(obj, measurements)
            N = size(measurements, 2);
            nx = length(obj.x0);
            x_est = zeros(nx, N); x_est(:, 1) = obj.x0;
            P = obj.P0; P_hist = zeros(nx, nx, N); P_hist(:, :, 1) = P;
            I = eye(nx);
            
            for k = 2:N
                x_prev = x_est(:, k-1);
                x_pred = obj.Model.F(x_prev);
                F_jac = obj.Model.dFdX(x_prev);
                P_pred = F_jac * P * F_jac' + obj.Q;
                
                z_meas = measurements(:, k);
                z_pred = obj.Model.H(x_pred);
                H_jac = obj.Model.dHdX(x_pred);
                
                y_res = z_meas - z_pred;
                S = H_jac * P_pred * H_jac' + obj.R;
                K = P_pred * H_jac' / S;
                
                x_est(:, k) = x_pred + K * y_res;
                P = (I - K * H_jac) * P_pred;
                P_hist(:, :, k) = P;
            end
        end

        %% 2b. IEKF (Iterated Extended Kalman Filter)
        function [x_est, P_hist] = runIEKF(obj, measurements, max_iter, tol)
            % IEKF provádí iterativní upřesnění měřicího kroku.
            % Vhodné pro silně nelineární měřicí funkce H(x).
            if nargin < 3, max_iter = 10; end
            if nargin < 4, tol = 1e-6; end
            
            N = size(measurements, 2);
            nx = length(obj.x0);
            x_est = zeros(nx, N); x_est(:, 1) = obj.x0;
            P = obj.P0; P_hist = zeros(nx, nx, N); P_hist(:, :, 1) = P;
            I = eye(nx);
            
            for k = 2:N
                % --- 1. Predikce (stejná jako u EKF) ---
                x_prev = x_est(:, k-1);
                x_pred = obj.Model.F(x_prev);
                F_jac = obj.Model.dFdX(x_prev);
                P_pred = F_jac * P * F_jac' + obj.Q;
                
                % --- 2. Iterativní Korekce (IEKF Update) ---
                z_meas = measurements(:, k);
                eta = x_pred; % Počáteční odhad pro iterace je predikovaný stav
                
                for i = 1:max_iter
                    H_jac = obj.Model.dHdX(eta);
                    z_eta = obj.Model.H(eta);
                    
                    % Výpočet inovační matice a zisku s aktuálním eta
                    S = H_jac * P_pred * H_jac' + obj.R;
                    K = P_pred * H_jac' / S;
                    
                    % Update odhadu stavu (Gauss-Newton krok)
                    % Vzorec zohledňuje rozdíl mezi x_pred a aktuálním eta
                    eta_next = x_pred + K * (z_meas - z_eta - H_jac * (x_pred - eta));
                    
                    % Kontrola konvergence
                    if norm(eta_next - eta) < tol
                        eta = eta_next;
                        break; 
                    end
                    eta = eta_next;
                end
                
                % Finální update s nejlepším eta
                H_jac_final = obj.Model.dHdX(eta);
                S_final = H_jac_final * P_pred * H_jac_final' + obj.R;
                K_final = P_pred * H_jac_final' / S_final;
                
                x_est(:, k) = eta;
                P = (I - K_final * H_jac_final) * P_pred;
                P_hist(:, :, k) = P;
            end
        end
        
        %% 3. UKF (Unscented Kalman Filter)
        function [x_est, P_hist] = runUKF(obj, measurements, alpha, beta, kappa)
            if nargin < 3, alpha = 1e-3; end
            if nargin < 4, beta = 2; end
            if nargin < 5, kappa = 0; end
            
            nx = length(obj.x0); nz = size(obj.R, 1); N = size(measurements, 2);
            lambda = alpha^2 * (nx + kappa) - nx;
            n_sigma = 2 * nx + 1;
            Wm = zeros(1, n_sigma); Wc = zeros(1, n_sigma);
            Wm(1) = lambda / (nx + lambda); Wc(1) = Wm(1) + (1 - alpha^2 + beta);
            Wm(2:end) = 1 / (2 * (nx + lambda)); Wc(2:end) = 1 / (2 * (nx + lambda));
            
            x_est = zeros(nx, N); x_est(:, 1) = obj.x0;
            P = obj.P0; P_hist = zeros(nx, nx, N); P_hist(:, :, 1) = P;
            
            for k = 2:N
                X_sigma = obj.generateSigmaPoints(x_est(:, k-1), P, nx, lambda);
                X_sigma_pred = zeros(nx, n_sigma);
                for i = 1:n_sigma, X_sigma_pred(:, i) = obj.Model.F(X_sigma(:, i)); end
                x_pred = sum(Wm .* X_sigma_pred, 2);
                P_pred = obj.Q;
                for i = 1:n_sigma, d = X_sigma_pred(:, i) - x_pred; P_pred = P_pred + Wc(i) * (d * d'); end
                
                X_sigma_upd = obj.generateSigmaPoints(x_pred, P_pred, nx, lambda);
                Z_sigma = zeros(nz, n_sigma);
                for i = 1:n_sigma, Z_sigma(:, i) = obj.Model.H(X_sigma_upd(:, i)); end
                z_pred = sum(Wm .* Z_sigma, 2);
                S = obj.R; Pxz = zeros(nx, nz);
                for i = 1:n_sigma
                    dz = Z_sigma(:, i) - z_pred; dx = X_sigma_upd(:, i) - x_pred;
                    S = S + Wc(i) * (dz * dz'); Pxz = Pxz + Wc(i) * (dx * dz');
                end
                K = Pxz / S;
                x_est(:, k) = x_pred + K * (measurements(:, k) - z_pred);
                P = P_pred - K * S * K';
                P_hist(:, :, k) = P;
            end
        end
        
        %% 4. RTSS (Rauch-Tung-Striebel Smoother)
        function [x_smooth, P_smooth] = runRTSS(obj, x_filt, P_filt, F_override)
            N = size(x_filt, 2); nx = length(obj.x0);
            x_smooth = zeros(nx, N); 
            P_smooth = zeros(nx, nx, N);
            x_smooth(:, N) = x_filt(:, N);
            P_smooth(:, :, N) = P_filt(:, :, N);
            
            use_const_F = false;
            F_const = [];
            if nargin > 3 && ~isempty(F_override)
                F_const = F_override;
                use_const_F = true;
            elseif isfield(obj.Model, 'F') && isnumeric(obj.Model.F)
                F_const = obj.Model.F; 
                use_const_F = true;
            end
            
            for k = N-1:-1:1
                x_k = x_filt(:, k); P_k = P_filt(:, :, k);
                if use_const_F
                    F = F_const;
                    x_pred_next = F * x_k;
                else
                    F = obj.Model.dFdX(x_k);
                    x_pred_next = obj.Model.F(x_k);
                end
                P_pred_next = F * P_k * F' + obj.Q;
                C = P_k * F' / P_pred_next;
                x_smooth(:, k) = x_k + C * (x_smooth(:, k+1) - x_pred_next);
                P_smooth(:, :, k) = P_k + C * (P_smooth(:, :, k+1) - P_pred_next) * C';
            end
        end
    end
    
    methods (Access = private)
        function X_sigma = generateSigmaPoints(~, x, P, nx, lambda)
            try sqrt_P = chol((nx + lambda) * P, 'lower');
            catch, [v, d] = eig((nx + lambda) * P); d(d<0)=1e-6; sqrt_P=v*sqrt(d)*v'; end
            X_sigma = [x, x + sqrt_P, x - sqrt_P];
        end
    end
end