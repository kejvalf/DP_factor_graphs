%% ========================================================================
%  MONTE CARLO SIMULACE - POROVNÁNÍ FG (Window), EKF a UKF
% =========================================================================
clc; clear; close all;

% --- NASTAVENÍ SIMULACE ---
N_mc = 1000;             % Počet Monte-Carlo běhů (pro ladění stačí méně, např. 50)
RMSE_THRESHOLD = 100;  % Limit pro úspěšný běh [m]
WINDOW_SIZE = 10;      % Velikost posuvného okna pro FG

% Ukládání výsledků (rozšířená struktura)
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
    sqrt(x(3)^2 + x(4)^2)];
model.dHdX = @(x) [dHdx_map(x(2), x(1)), dHdy_map(x(2), x(1)), 0                          ,0;
    0                 , 0                   ,x(3) / (sqrt(x(3)^2 + x(4)^2)), x(4) / (sqrt(x(3)^2 + x(4)^2))];

% Dynamika
syms x_sym [model.nx 1] real
dt_sym = sym(model.dt);
F_sym = [x_sym(1) + dt_sym * x_sym(3);
    x_sym(2) + dt_sym * x_sym(4);
    x_sym(3);
    x_sym(4)];
dFdX_sym = jacobian(F_sym, x_sym);
model.F = matlabFunction(F_sym, 'Vars', {x_sym});
model.dFdX = matlabFunction(dFdX_sym, 'Vars', {x_sym});

% Matice šumu
q = 0.1;
Q = q * [ (model.dt^3/3) 0 (model.dt^2/2) 0;
    0 (model.dt^3/3) 0 (model.dt^2/2);
    (model.dt^2/2) 0 model.dt 0;
    0 (model.dt^2/2) 0 model.dt ];
stdV = 0.1;
R = diag([0.5^2, stdV^2]);

% Ground Truth
poloha = souradniceGNSS(1:2, :);
N_total = size(poloha, 2);
rychlost = [diff(poloha, 1, 2), [0;0]];
x_true_all = [poloha; rychlost];

% ========================================================================
%  MONTE CARLO SMYČKA
% =========================================================================
fprintf('Spouštím Monte Carlo simulaci (%d iterací)...\n', N_mc);
wb = waitbar(0, 'Probíhá Monte Carlo simulace...');

for i_mc = 1:N_mc

    current_seed = i_mc; % zatim
    rng(current_seed);

    % Data storage structure
    mc_results(i_mc).iteration = i_mc;
    mc_results(i_mc).seed = current_seed;

    try
        % -----------------------------------------------------------
        % 1. Generování měření (unikátní šum pro každý běh)
        % -----------------------------------------------------------
        meas_all = zeros(model.nz, N_total);
        for k = 1:N_total
            xk = x_true_all(:, k);
            % Teoretické měření + šum - aktualizované
            meas_clean = [H_map(xk(2),xk(1)); %hB(k)
                          sqrt(xk(3)^2+xk(4)^2)];
            % Přidání šumu
            meas_all(:, k) = meas_clean + sqrt(diag(R)) .* randn(model.nz,1);
        end

        % Inicializace Priori
        P0 = diag([1, 1, 1, 1]).^2;
        x0 = x_true_all(:,1) + sqrt(P0) * randn(4,1);

        % Vytvoření objektu filtru (pro EKF/UKF)
        myFilter = TrajectoryFilters(model, Q, R, x0, P0);

        % -----------------------------------------------------------
        % 2. EKF Estimace
        % -----------------------------------------------------------
        tic;
        try
            [x_est_EKF, ~] = myFilter.runEKF(meas_all);

            % Výpočet RMSE (stejný rozsah jako FG: od WINDOW_SIZE do konce)
            diff_ekf = x_est_EKF(1:2, WINDOW_SIZE:end) - x_true_all(1:2, WINDOW_SIZE:end);
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
        % Reset filtru nebo vytvoření nového pro čistý start UKF
        myFilterUKF = TrajectoryFilters(model, Q, R, x0, P0);
        tic;
        try
            [x_est_UKF, ~] = myFilterUKF.runUKF(meas_all);

            diff_ukf = x_est_UKF(1:2, WINDOW_SIZE:end) - x_true_all(1:2, WINDOW_SIZE:end);
            rmse_ukf = sqrt(mean(sum(diff_ukf.^2, 1)));

            mc_results(i_mc).rmse_ukf = rmse_ukf;
            mc_results(i_mc).success_ukf = (rmse_ukf < RMSE_THRESHOLD);
        catch
            mc_results(i_mc).rmse_ukf = NaN;
            mc_results(i_mc).success_ukf = false;
        end
        mc_results(i_mc).time_ukf = toc;

        % -----------------------------------------------------------
        % 4. FG Sliding Window Estimace
        % -----------------------------------------------------------
        tic;
        try
            estimated_states_history = zeros(model.nx, N_total);

            % A) První okno
            meas_window = meas_all(:, 1:WINDOW_SIZE);
            solver = FactorGraphSolver_window(model, meas_window, R, Q, x0, P0);
            solver.opt();

            current_states = solver.states';
            estimated_states_history(:, 1:WINDOW_SIZE) = current_states;
            P_full = solver.compute_covariance();

            % B) Hlavní smyčka
            for k = WINDOW_SIZE + 1 : N_total
                indices = (k - WINDOW_SIZE + 1) : k;
                meas_window = meas_all(:, indices);

                % Marginalizace
                x0_new = current_states(:, 2);
                idx_cov = model.nx + (1:model.nx);
                P0_new = P_full(idx_cov, idx_cov);

                % Warm Start
                prev_trajectory_shifted = current_states(:, 2:end);
                last_state = prev_trajectory_shifted(:, end);
                new_prediction = model.F(last_state);
                initial_guess = [prev_trajectory_shifted, new_prediction]';

                % Update & Opt
                solver.update_problem(meas_window, x0_new, P0_new, initial_guess);
                solver.opt();

                current_states = solver.states';
                estimated_states_history(:, k) = current_states(:, end);
                P_full = solver.compute_covariance();
            end

            % Vyhodnocení FG
            diff_fg = estimated_states_history(1:2, WINDOW_SIZE:end) - x_true_all(1:2, WINDOW_SIZE:end);
            rmse_fg = sqrt(mean(sum(diff_fg.^2, 1)));

            mc_results(i_mc).rmse_fg = rmse_fg;
            mc_results(i_mc).success_fg = (rmse_fg < RMSE_THRESHOLD);

        catch
            mc_results(i_mc).rmse_fg = NaN;
            mc_results(i_mc).success_fg = false;
        end
        mc_results(i_mc).time_fg = toc;

    catch ME
        fprintf('Kritická chyba v iteraci %d: %s\n', i_mc, ME.message);
    end

    waitbar(i_mc/N_mc, wb, sprintf('MC Iterace %d/%d', i_mc, N_mc));
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

fprintf('%-10s | %-4d (%5.1f%%)    | %-12.4f\n', 'FG Window', ...
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
title(['FG Window (Průměr: ' num2str(mean(rmse_fg_valid), '%.2f') ' m)']);
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