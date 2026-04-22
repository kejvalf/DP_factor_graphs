%% ========================================================================
%  MONTE CARLO SIMULACE - Vliv velikosti okna u FG vs EKF a UKF
% =========================================================================
clc; clear; close all;
% --- NASTAVENÍ SIMULACE ---
N_mc = 100;             % Počet Monte-Carlo běhů
RMSE_THRESHOLD = 500;   % Limit pro úspěšný běh [m]

% Definice délek oken, které chceme testovat
WINDOW_SIZES = linspace(2,40,39); % Příklad délek oken
num_windows = length(WINDOW_SIZES);

% --- PREALOKACE VÝSLEDKŮ ---
% Pro EKF a UKF
rmse_ekf_all = NaN(N_mc, 1);  time_ekf_all = NaN(N_mc, 1);
rejected_ekf_all = false(N_mc, 1); % Sledování úspěšnosti EKF

rmse_ukf_all = NaN(N_mc, 1);  time_ukf_all = NaN(N_mc, 1);
rejected_ukf_all = false(N_mc, 1); % Sledování úspěšnosti UKF

% Pro FG (matice: řádky = MC běhy, sloupce = velikosti oken)
rmse_fg_all  = NaN(N_mc, num_windows);
time_fg_all  = NaN(N_mc, num_windows);
rejected_fg_all = false(N_mc, num_windows); % Sledování úspěšnosti FG

% --- DEFINICE MODELU (Statická část) ---
model.nx = 4; % [x, y, vx, vy]
model.nz = 2; % [z_baro, pseudo_vx, pseudo_vy]
model.dt = 1.0;

% Načtení dat (předpokládá se existence souboru data.mat)
load("data.mat"); 

% Příprava mapy
xv = souradniceX(1,:);
yv = souradniceY(:,1);
[map_m.x,map_m.y] = meshgrid(xv,yv);
map_m.z = souradniceZ;

[Gx, Gy] = gradient(map_m.z, mean(diff(xv)), mean(diff(yv)));
H_map    = griddedInterpolant({yv,xv}, map_m.z, 'linear','nearest');
dHdx_map = griddedInterpolant({yv,xv}, Gx, 'linear','nearest');
dHdy_map = griddedInterpolant({yv,xv}, Gy, 'linear','nearest');

model.H = @(x)[H_map(x(2), x(1));
    sqrt(x(3)^2 + x(4)^2)];
model.dHdX = @(x)[dHdx_map(x(2), x(1)), dHdy_map(x(2), x(1)), 0                          ,0;
    0                 , 0                   ,x(3) / (sqrt(x(3)^2 + x(4)^2)), x(4) / (sqrt(x(3)^2 + x(4)^2))];

% Dynamika
syms x_sym [model.nx 1] real
dt_sym = sym(model.dt);
F_sym =[x_sym(1) + dt_sym * x_sym(3);
    x_sym(2) + dt_sym * x_sym(4);
    x_sym(3);
    x_sym(4)];
dFdX_sym = jacobian(F_sym, x_sym);
model.F = matlabFunction(F_sym, 'Vars', {x_sym});
model.dFdX = matlabFunction(dFdX_sym, 'Vars', {x_sym});

% Matice šumu
q = 0.1;
Q = q *[ (model.dt^3/3) 0 (model.dt^2/2) 0;
    0 (model.dt^3/3) 0 (model.dt^2/2);
    (model.dt^2/2) 0 model.dt 0;
    0 (model.dt^2/2) 0 model.dt ];
stdV = 0.1;
R = diag([0.5^2, stdV^2]);

% Ground Truth
poloha = souradniceGNSS(1:2, 1:400);
N_total = size(poloha, 2);
rychlost = [diff(poloha, 1, 2), [0;0]];
x_true_all = [poloha; rychlost];

EVAL_START_IDX = 1; 

% ========================================================================
%  MONTE CARLO SMYČKA
% =========================================================================
fprintf('Spouštím Monte Carlo simulaci (%d iterací, max okno %d)...\n', N_mc, max(WINDOW_SIZES));
wb = waitbar(0, 'Probíhá Monte Carlo simulace...');

for i_mc = 1:N_mc
    rng(i_mc); 

    % Generování měření pro tento běh
    meas_all = zeros(model.nz, N_total);
    for k = 1:N_total
        xk = x_true_all(:, k);
        meas_clean =[H_map(xk(2),xk(1)); sqrt(xk(3)^2+xk(4)^2)];
        meas_all(:, k) = meas_clean + sqrt(diag(R)) .* randn(model.nz,1);
    end

    % Inicializace Priori
    P0 = diag([10, 10, 0.5, 0.5]).^2;
    x0 = x_true_all(:,1) + sqrt(P0) * randn(4,1);

    % -----------------------------------------------------------
    % 2. EKF Estimace
    % -----------------------------------------------------------
    myFilter = TrajectoryFilters(model, Q, R, x0, P0);
    tic;
    try
        [x_est_EKF, ~] = myFilter.runEKF(meas_all);
        diff_ekf = x_est_EKF(1:4, EVAL_START_IDX:end) - x_true_all(1:4, EVAL_START_IDX:end);
        rmse_val = sum(sqrt(mean(diff_ekf.^2, 2)), 1);
        rmse_ekf_all(i_mc) = rmse_val;
        
        % Kontrola úspěšnosti
        if isnan(rmse_val) || rmse_val > RMSE_THRESHOLD
            rejected_ekf_all(i_mc) = true;
        end
    catch
        rejected_ekf_all(i_mc) = true;
    end
    time_ekf_all(i_mc) = toc;

    % -----------------------------------------------------------
    % 3. UKF Estimace
    % -----------------------------------------------------------
    myFilterUKF = TrajectoryFilters(model, Q, R, x0, P0);
    tic;
    try
        [x_est_UKF, ~] = myFilterUKF.runUKF(meas_all);
        diff_ukf = x_est_UKF(1:4, EVAL_START_IDX:end) - x_true_all(1:4, EVAL_START_IDX:end);
        rmse_val = sum(sqrt(mean(diff_ukf.^2, 2)), 1);
        rmse_ukf_all(i_mc) = rmse_val;
        
        if isnan(rmse_val) || rmse_val > RMSE_THRESHOLD
            rejected_ukf_all(i_mc) = true;
        end
    catch
        rejected_ukf_all(i_mc) = true;
    end
    time_ukf_all(i_mc) = toc;

    % -----------------------------------------------------------
    % 4. FG Sliding Window Estimace
    % -----------------------------------------------------------
    for w_idx = 1:num_windows
        W = WINDOW_SIZES(w_idx);
        tic;
        try
            estimated_states_history = zeros(model.nx, N_total);
            meas_window = meas_all(:, 1:W);
            solver = FactorGraphSolver_window(model, meas_window, R, Q, x0, P0);
            solver.opt();
            current_states = solver.states';
            estimated_states_history(:, 1:W) = current_states;
            P_full = solver.compute_covariance();

            for k = W + 1 : N_total
                indices = (k - W + 1) : k;
                meas_window_k = meas_all(:, indices);
                x0_new = current_states(:, 2);
                idx_cov = model.nx + (1:model.nx);
                P0_new = P_full(idx_cov, idx_cov);
                prev_trajectory_shifted = current_states(:, 2:end);
                last_state = prev_trajectory_shifted(:, end);
                new_prediction = model.F(last_state);
                initial_guess = [prev_trajectory_shifted, new_prediction]';
                solver.update_problem(meas_window_k, x0_new, P0_new, initial_guess);
                solver.opt();
                current_states = solver.states';
                estimated_states_history(:, k) = current_states(:, end);
                P_full = solver.compute_covariance();
            end

            diff_fg = estimated_states_history(1:4, EVAL_START_IDX:end) - x_true_all(1:4, EVAL_START_IDX:end);
            rmse_val = sum(sqrt(mean(diff_fg.^2, 2)), 1);
            rmse_fg_all(i_mc, w_idx) = rmse_val;

            if isnan(rmse_val) || rmse_val > RMSE_THRESHOLD
                rejected_fg_all(i_mc, w_idx) = true;
            end
        catch
            rejected_fg_all(i_mc, w_idx) = true;
        end
        time_fg_all(i_mc, w_idx) = toc;
    end 

    waitbar(i_mc/N_mc, wb, sprintf('MC Iterace %d/%d', i_mc, N_mc));
end
close(wb);

% ========================================================================
%  VYHODNOCENÍ VÝSLEDKŮ - STATISTIKY ZAMÍTNUTÍ
% =========================================================================

% Výpočet procenta zamítnutých běhů
pct_rejected_ekf = (sum(rejected_ekf_all) / N_mc) * 100;
pct_rejected_ukf = (sum(rejected_ukf_all) / N_mc) * 100;
pct_rejected_fg  = (sum(rejected_fg_all, 1) / N_mc) * 100;

fprintf('\n--- STATISTIKY ÚSPĚŠNOSTI (Threshold: %d m) ---\n', RMSE_THRESHOLD);
fprintf('EKF zamítnuto: %.1f %%\n', pct_rejected_ekf);
fprintf('UKF zamítnuto: %.1f %%\n', pct_rejected_ukf);
for w_idx = 1:num_windows
    fprintf('FG (okno %d) zamítnuto: %.1f %%\n', WINDOW_SIZES(w_idx), pct_rejected_fg(w_idx));
end

% Výpočet průměrů pouze z úspěšných běhů
rmse_ekf_filtered = rmse_ekf_all(~rejected_ekf_all);
avg_rmse_ekf = mean(rmse_ekf_filtered, 'omitnan');
avg_time_ekf = mean(time_ekf_all, 'omitnan');

rmse_ukf_filtered = rmse_ukf_all(~rejected_ukf_all);
avg_rmse_ukf = mean(rmse_ukf_filtered, 'omitnan');
avg_time_ukf = mean(time_ukf_all, 'omitnan');

rmse_fg_filtered = rmse_fg_all;
rmse_fg_filtered(rejected_fg_all) = NaN; % Vynulování neúspěšných pro průměr
avg_rmse_fg = mean(rmse_fg_filtered, 1, 'omitnan');
avg_time_fg = mean(time_fg_all, 1, 'omitnan');

%% ========================================================================
%  VYKRESLENÍ GRAFŮ - Monte Carlo Analýza
% =========================================================================

fig_mc = figure('Name', 'Analýza MC', 'Color', 'w', 'Position',[100, 100, 900, 800]);

% Vytvoření mřížky 3x1 s minimálními mezerami
t_layout = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Určení jednotného rozsahu osy X pro všechny 3 grafy
xlim_range = [min(WINDOW_SIZES) - 1, max(WINDOW_SIZES) + 1];

% --- 1. RMSE Graf ---
nexttile;
hold on; grid on;
% Uložíme si handly čar pro pozdější legendu
h_fg = plot(WINDOW_SIZES, avg_rmse_fg, 'b-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'w');
h_ekf = yline(avg_rmse_ekf, '--', 'Color', '#EDB120', 'LineWidth', 1.5);
h_ukf = yline(avg_rmse_ukf, '-.', 'Color', 'r', 'LineWidth', 1.5);

ylabel('Průměrné RMSE [m]', 'Interpreter', 'tex', 'FontSize', 13);
title('\bf Přesnost (pouze úspěšné běhy)', 'Interpreter', 'tex', 'FontSize', 14);
xlim(xlim_range);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
xticklabels({}); % Skryje čísla na ose X

% --- 2. Procento zamítnutí ---
nexttile;
hold on; grid on;
% Barva sloupce změněna na modro-šedou, aby vizuálně odpovídala FG (který je modrý)
bar(WINDOW_SIZES, pct_rejected_fg, 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'k');
yline(pct_rejected_ekf, '--', 'Color', '#EDB120', 'LineWidth', 1.5);
yline(pct_rejected_ukf, '-.', 'Color', 'r', 'LineWidth', 1.5);

ylabel('Zamítnuto [%]', 'Interpreter', 'tex', 'FontSize', 13);
title(sprintf('\\bf Spolehlivost (procento běhů s RMSE > %d m)', RMSE_THRESHOLD), 'Interpreter', 'tex', 'FontSize', 14);
xlim(xlim_range);
ylim([0 105]);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
xticklabels({}); % Skryje čísla na ose X

% --- 3. Výpočetní čas ---
nexttile;
hold on; grid on;
plot(WINDOW_SIZES, avg_time_fg, 'b-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'w');
yline(avg_time_ekf, '--', 'Color', '#EDB120', 'LineWidth', 1.5);
yline(avg_time_ukf, '-.', 'Color', 'r', 'LineWidth', 1.5);

xlabel('Délka okna {\it N} [kroky]', 'Interpreter', 'tex', 'FontSize', 14);
ylabel('Čas [s]', 'Interpreter', 'tex', 'FontSize', 13);
title('\bf Výpočetní náročnost', 'Interpreter', 'tex', 'FontSize', 14);
xlim(xlim_range);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;

% --- Hlavní nadpis a sdílená legenda ---
title(t_layout, sprintf('\\bf Monte Carlo analýza vlivu délky okna (%d iterací)', N_mc), 'Interpreter', 'tex', 'FontSize', 16);

% Společná legenda vložená bezpečně dolů pomocí Tile = 'south'
lgd = legend([h_fg, h_ekf, h_ukf], 'FG (Klouzavé okno)', 'EKF', 'UKF', ...
    'Interpreter', 'tex', 'Orientation', 'horizontal');
lgd.FontSize = 12;
lgd.Layout.Tile = 'south'; % Umístí legendu pod osu X třetího grafu

exportgraphics(fig_mc, 'obrazky/TAN/MC_delka_okna_RMSE.pdf', 'ContentType', 'vector');