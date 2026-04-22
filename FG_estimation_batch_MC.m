%% ========================================================================
%  MONTE CARLO SIMULACE - BATCH FG vs EKF vs UKF
% =========================================================================
clc; clear; close all;

% --- NASTAVENÍ SIMULACE ---
N_mc = 1000;             % Počet Monte-Carlo běhů
RMSE_THRESHOLD = 100;  % Limit pro úspěšný běh (metry)
BATCH_SIZE = 1276;      % Velikost dávky pro FG Batch

% Ukládání výsledků
mc_results = struct('iteration', [], 'seed', [], ...
                    'rmse_fg', [], 'time_fg', [], 'success_fg', [], ...
                    'rmse_ekf', [], 'time_ekf', [], 'success_ekf', [], ...
                    'rmse_ukf', [], 'time_ukf', [], 'success_ukf', []);

% --- DEFINICE MODELU (Statická část) ---
model.nx = 4; % [x, y, vx, vy]
model.nz = 3; % [z_baro, pseudo_vx, pseudo_vy]
model.dt = 1.0; 

load("data.mat"); % Očekává: souradniceGNSS, souradniceX, souradniceY, souradniceZ, hB

% Příprava mapy
xv = souradniceX(1,:);
yv = souradniceY(:,1);
[map_m.x,map_m.y] = meshgrid(xv,yv);
map_m.z = souradniceZ;

[Gx, Gy] = gradient(map_m.z, mean(diff(xv)), mean(diff(yv)));
H_map    = griddedInterpolant({yv,xv}, map_m.z, 'linear','nearest');
dHdx_map = griddedInterpolant({yv,xv}, Gx, 'linear','nearest');
dHdy_map = griddedInterpolant({yv,xv}, Gy, 'linear','nearest');

model.H = @(x) [H_map(x(2), x(1));
                (x(3)./sqrt(x(3).^2+x(4).^2)).*x(3)-(x(4)./sqrt(x(3).^2+x(4).^2)).*x(4);
                (x(4)./sqrt(x(3).^2+x(4).^2)).*x(3)+(x(3)./sqrt(x(3).^2+x(4).^2)).*x(4)];

model.dHdX = @(x) [dHdx_map(x(2), x(1)), dHdy_map(x(2), x(1)), 0, 0;
                   0, 0, (x(3)^3 + 3*x(3)*x(4)^2)/sqrt((x(3)^2 + x(4)^2)^3), -(3*x(3)^2*x(4) + x(4)^3)/sqrt((x(3)^2 + x(4)^2)^3);
                   0, 0, 2*x(4)^3/sqrt((x(3)^2 + x(4)^2)^3),                 2*x(3)^3/sqrt((x(3)^2 + x(4)^2)^3)];

% Dynamika
syms x_sym [model.nx 1] real
dt_sym = sym(model.dt);
model.F_sym = [x_sym(1) + dt_sym * x_sym(3);
               x_sym(2) + dt_sym * x_sym(4);
               x_sym(3);
               x_sym(4)];
model.dFdX_sym = jacobian(model.F_sym, x_sym);
model.F = matlabFunction(model.F_sym, 'Vars', {x_sym});
model.dFdX = matlabFunction(model.dFdX_sym, 'Vars', {x_sym});

% Matice šumu
q = 0.1;
Q = q * [ (model.dt^3/3) 0 (model.dt^2/2) 0;
          0 (model.dt^3/3) 0 (model.dt^2/2);
          (model.dt^2/2) 0 model.dt 0;
          0 (model.dt^2/2) 0 model.dt ];
stdV = 0.5; 
R = diag([5^2, stdV^2, stdV^2]);

% Ground Truth (Fixní délka)
data_range = 1:size(souradniceGNSS,2); % Délka simulace
souradniceGNSS = souradniceGNSS(:, data_range);
hB = hB(:, data_range);
poloha = souradniceGNSS(1:2, :);
rychlost = [diff(poloha, 1, 2), [0;0]];
rychlost(:, end) = rychlost(:, end-1); 
x_true_all = [poloha; rychlost];
N_total = size(x_true_all, 2);

num_batches = ceil(N_total / BATCH_SIZE); 

% ========================================================================
%  MONTE CARLO SMYČKA
% =========================================================================
fprintf('Spouštím Monte Carlo (Batch vs Filtry, %d iterací)...\n', N_mc);
wb = waitbar(0, 'Probíhá Monte Carlo simulace...');

for i_mc = 1:N_mc
    
    current_seed = i_mc; 
    rng(current_seed); 
    
    mc_results(i_mc).iteration = i_mc;
    mc_results(i_mc).seed = current_seed;

    try
        % -----------------------------------------------------------
        % 1. Generování měření
        % -----------------------------------------------------------
        meas_all = zeros(model.nz, N_total);
        for k = 1:N_total
            xk = x_true_all(:, k);
            v_norm = sqrt(xk(3)^2 + xk(4)^2); 
            if v_norm == 0, v_norm = eps; end
            
            meas_clean = [hB(k);
                          (xk(3)/v_norm)*xk(3) - (xk(4)/v_norm)*xk(4);
                          (xk(4)/v_norm)*xk(3) + (xk(3)/v_norm)*xk(4)];
            
            meas_all(:, k) = meas_clean + sqrt(diag(R)) .* randn(3,1);
            meas_all(1, k) = hB(k); 
        end
        
        % Inicializace Priori
        P0 = diag([1, 1, 1, 1]).^2;
        x0 = x_true_all(:,1) + sqrt(P0) * randn(4,1);

        % -----------------------------------------------------------
        % 2. EKF Estimace
        % -----------------------------------------------------------
        tic;
        try
            myFilterEKF = TrajectoryFilters(model, Q, R, x0, P0);
            [x_est_EKF, ~] = myFilterEKF.runEKF(meas_all);
            
            diff_ekf = x_est_EKF(1:2, :) - x_true_all(1:2, :);
            rmse_ekf = sqrt(mean(sum(diff_ekf.^2, 1)));
            
            mc_results(i_mc).rmse_ekf = rmse_ekf;
            mc_results(i_mc).success_ekf = (rmse_ekf < RMSE_THRESHOLD);
        catch
            mc_results(i_mc).rmse_ekf = NaN;
            mc_results(i_mc).success_ekf = false;
        end
        mc_results(i_mc).time_ekf = toc;

        % -----------------------------------------------------------
        % 3. UKF Estimace
        % -----------------------------------------------------------
        tic;
        try
            myFilterUKF = TrajectoryFilters(model, Q, R, x0, P0);
            [x_est_UKF, ~] = myFilterUKF.runUKF(meas_all);
            
            diff_ukf = x_est_UKF(1:2, :) - x_true_all(1:2, :);
            rmse_ukf = sqrt(mean(sum(diff_ukf.^2, 1)));
            
            mc_results(i_mc).rmse_ukf = rmse_ukf;
            mc_results(i_mc).success_ukf = (rmse_ukf < RMSE_THRESHOLD);
        catch
            mc_results(i_mc).rmse_ukf = NaN;
            mc_results(i_mc).success_ukf = false;
        end
        mc_results(i_mc).time_ukf = toc;

        % -----------------------------------------------------------
        % 4. FG Batch Estimace
        % -----------------------------------------------------------
        tic;
        try
            estimated_states_fg = zeros(model.nx, N_total);
            
            % Inicializace pro první batch (reset proměnných)
            x0_batch = x0;
            P0_batch = P0;
            
            % Vytvoření solveru s dummy daty (recyklace objektu)
            meas_dummy = meas_all(:, 1:min(BATCH_SIZE, N_total));
            solver = FactorGraphSolver(model, meas_dummy, R, Q, x0_batch, P0_batch);
            
            current_idx = 1; 
            
            for k = 1:num_batches
                % Výběr dat
                idx_start = (k-1) * BATCH_SIZE + 1;
                idx_end   = min(k * BATCH_SIZE, N_total);
                indices   = idx_start:idx_end;
                meas_batch = meas_all(:, indices);
                
                % Update & Opt
                solver.update_problem(meas_batch, x0_batch, P0_batch);
                solver.opt();
                
                % Uložení výsledků
                est_state_batch = solver.states';
                n_samples = size(est_state_batch, 2);
                idx_range = current_idx : (current_idx + n_samples - 1);
                estimated_states_fg(:, idx_range) = est_state_batch;
                
                % Stitching (Navázání)
                if k < num_batches
                    x_end = est_state_batch(:, end);
                    
                    % Vytažení poslední kovariance pro predikci
                    P_full_batch = solver.compute_covariance();
                    idx_s = (n_samples-1)*model.nx + 1;
                    idx_e = n_samples*model.nx;
                    P_end = full(P_full_batch(idx_s:idx_e, idx_s:idx_e));
                    
                    % Predikce do začátku dalšího batche
                    x0_batch = model.F(x_end);
                    F_jac = model.dFdX(x_end);
                    P0_batch = F_jac * P_end * F_jac' + Q;
                    P0_batch = (P0_batch + P0_batch') / 2;
                    
                    current_idx = current_idx + n_samples;
                end
            end
            
            % Vyhodnocení FG
            diff_fg = estimated_states_fg(1:2, :) - x_true_all(1:2, :);
            rmse_fg = sqrt(mean(sum(diff_fg.^2, 1)));
            
            mc_results(i_mc).rmse_fg = rmse_fg;
            mc_results(i_mc).success_fg = (rmse_fg < RMSE_THRESHOLD);

        catch
            mc_results(i_mc).rmse_fg = NaN;
            mc_results(i_mc).success_fg = false;
        end
        mc_results(i_mc).time_fg = toc;

    catch ME
        fprintf('Kritická chyba iterace %d: %s\n', i_mc, ME.message);
    end
    
    waitbar(i_mc/N_mc, wb, sprintf('Iterace %d/%d', i_mc, N_mc));
end
close(wb);

% ========================================================================
%  VYHODNOCENÍ - TABULKA A HISTOGRAMY
% =========================================================================

% 1. Příprava dat (filtrace platných RMSE pro výpočet průměru)
rmse_fg_valid  = [mc_results.rmse_fg]; rmse_fg_valid  = rmse_fg_valid(~isnan(rmse_fg_valid));
rmse_ukf_valid = [mc_results.rmse_ukf]; rmse_ukf_valid = rmse_ukf_valid(~isnan(rmse_ukf_valid));
rmse_ekf_valid = [mc_results.rmse_ekf]; rmse_ekf_valid = rmse_ekf_valid(~isnan(rmse_ekf_valid));

% Průměrné časy
time_fg_avg = mean([mc_results.time_fg], 'omitnan');
time_ekf_avg = mean([mc_results.time_ekf], 'omitnan');
time_ukf_avg = mean([mc_results.time_ukf], 'omitnan');

% 2. Výpočet počtů úspěšných běhů (pod prahem)
n_success_fg  = sum([mc_results.success_fg]);
n_success_ekf = sum([mc_results.success_ekf]);
n_success_ukf = sum([mc_results.success_ukf]);

% 3. Výpočet procent úspěšnosti
pct_fg  = (n_success_fg  / N_mc) * 100;
pct_ekf = (n_success_ekf / N_mc) * 100;
pct_ukf = (n_success_ukf / N_mc) * 100;

% 4. Vylepšená Tabulka výsledků
fprintf('\n======================================================================\n');
fprintf('VÝSLEDKY MC (%d iterací, Práh=%.0f m)\n', N_mc, RMSE_THRESHOLD);
fprintf('======================================================================\n');
% Formátování hlavičky: Metoda | RMSE | Počet (Procenta) | Čas
fprintf('%-10s | %-18s | %-12s\n', 'Metoda', 'Úspěšnost [N (%)]', 'Prům. Čas [s]');
fprintf('----------------------------------------------------------------------\n');

fprintf('%-10s | %-4d (%5.1f%%)    | %-12.4f\n', 'FG Batch', ...
    n_success_fg, pct_fg, time_fg_avg);

fprintf('%-10s | %-4d (%5.1f%%)    | %-12.4f\n', 'EKF', ...
    n_success_ekf, pct_ekf, time_ekf_avg);

fprintf('%-10s | %-4d (%5.1f%%)    | %-12.4f\n', 'UKF', ...
    n_success_ukf, pct_ukf, time_ukf_avg);

fprintf('======================================================================\n');

% 5. Histogramy (kód zůstává stejný nebo mírně upravený)
figure('Name', 'Histogramy RMSE', 'Color', 'w', 'Position', [100, 100, 600, 800]);

max_val_safe = max([max(rmse_fg_valid), max(rmse_ukf_valid), RMSE_THRESHOLD * 1.2]);
if ~isempty(rmse_ekf_valid) && max(rmse_ekf_valid) < (RMSE_THRESHOLD * 5)
    max_val_safe = max(max_val_safe, max(rmse_ekf_valid));
end
common_xlim = [0, max_val_safe];
n_bins = 20;

% --- SUBPLOT 1: FG Batch ---
subplot(3,1,1);
histogram(rmse_fg_valid, n_bins, 'FaceColor', 'b', 'EdgeColor', 'k', 'FaceAlpha', 0.6);
hold on; xline(RMSE_THRESHOLD, 'r--', 'LineWidth', 2, 'Label', 'Práh');
title(['FG Batch (Průměr: ' num2str(mean(rmse_fg_valid), '%.2f') ' m)']);
ylabel('Počet'); xlim(common_xlim); grid on;

% --- SUBPLOT 2: UKF ---
subplot(3,1,2);
histogram(rmse_ukf_valid, n_bins, 'FaceColor', 'r', 'EdgeColor', 'k', 'FaceAlpha', 0.6);
hold on; xline(RMSE_THRESHOLD, 'r--', 'LineWidth', 2, 'Label', 'Práh');
title(['UKF (Průměr: ' num2str(mean(rmse_ukf_valid), '%.2f') ' m)']);
ylabel('Počet'); xlim(common_xlim); grid on;

% --- SUBPLOT 3: EKF ---
subplot(3,1,3);
histogram(rmse_ekf_valid, n_bins, 'FaceColor', '#EDB120', 'EdgeColor', 'k', 'FaceAlpha', 0.6);
hold on; xline(RMSE_THRESHOLD, 'r--', 'LineWidth', 2, 'Label', 'Práh');
title(['EKF (Průměr: ' num2str(mean(rmse_ekf_valid), '%.2f') ' m)']);
xlabel('RMSE Polohy [m]'); ylabel('Počet'); xlim(common_xlim); grid on;