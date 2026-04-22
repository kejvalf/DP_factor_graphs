%% ========================================================================
% Spouštěcí skript pro FG estimaci - Linearni systém
% =========================================================================
clc; clear;
close all;

% ========================================================================
%  DEFINICE LINEARNIHO MODELU
% =========================================================================
model.nx = 3; % [x, vx, bias]
model.nz = 2; %
model.dt = 1.0;
% Funkce měření H
model.H = @(x) [x(1) + x(3);
    x(1)];
% Jacobián měření H
model.dHdX = @(x) [1, 0, 1;
    1, 0, 0];
% Dynamika  F
model.F = @(x) [x(1) + model.dt*x(2);
    x(2);
    x(3)];
% Jakobidán dynamiky  F
model.dFdX =@(x) [1 ,model.dt, 0;
    0 ,1       , 0;
    0 ,0       , 1];
% Matice šumu Q a R
q_pos  = 1e-1;
q_vel  = 1e-1;
q_bias = 1e-2;
Q = diag([q_pos, q_vel, q_bias]);
r1 = 1e-1;
r2 = 5e-1;
R = diag([r1, r2]);

%% PŘÍPRAVA DAT
% Generování měření pro celou trajektorii
load("vysledky_linear.csv") % k|true_x|true_v|true_vx|true_vy|meas1|meas2|meas3|meas4|est_x|est_y|est_vx|est_vy
N = size(vysledky_linear,1);

x0 = [0; 0; 0];                    % počáteční stav (x,y,vx,vy)
P0  = 0.001 * eye(3);
% prior variance
meas_all = zeros(model.nz, N);
x_true_all = zeros(model.nx, N+1);
% Stavy a měření se generují v C++ kódu, zde si je jen zkopíruji
x_true_all =vysledky_linear(:,2:4)';
meas_all = vysledky_linear(:,5:6)';

%% FG window vyhlazené odhady
WINDOW_SIZE = 10; % Velikost okna
fprintf('Spouštím Sliding Window Estimaci (okno = %d)...\n', WINDOW_SIZE);

% Pole pro výsledky
estimated_states_history = zeros(model.nx, N);
estimated_variance_history = cell(1, N);

% --- INICIALIZACE PRVNÍHO OKNA ---
meas_window = meas_all(:, 1:WINDOW_SIZE);

% Vytvoříme objekt solveru (jednou)
solver = FactorGraphSolver_window(model, meas_window, R, Q, x0, P0);
solver.opt();

% Uložení výsledků z prvního okna
current_states = solver.states'; % [nx x WINDOW_SIZE]
estimated_states_history(:, 1:WINDOW_SIZE) = current_states;
smooth_estimate{:,1} = current_states';


P_full = solver.compute_covariance();
for t = 1:WINDOW_SIZE
    idx = (t-1)*model.nx + (1:model.nx);
    estimated_variance_history{t} = P_full(idx, idx);
end

tic;
for k = WINDOW_SIZE + 1 : N

    % A) Příprava dat pro nové okno
    indices = (k - WINDOW_SIZE + 1) : k;
    meas_window = meas_all(:, indices);

    % B) Marginalizace (Posun Prioru)
    % Stav x_2 z minulého okna se stává x_1 (x0) nového okna
    x0_new = current_states(:, 2);

    % Extrakce kovariance pro tento stav z velké matice P
    % V minulém okně to byl 2. stav -> indexy (nx+1 až 2*nx)
    idx_cov = model.nx + (1:model.nx);
    P0_new = P_full(idx_cov, idx_cov);

    % C) Warm Start (Předvyplnění trajektorie)
    % Posuneme trajektorii o jedna doleva a predikujeme nový bod
    prev_trajectory_shifted = current_states(:, 2:end);
    last_state = prev_trajectory_shifted(:, end);
    new_prediction = model.F(last_state);

    initial_guess = [prev_trajectory_shifted, new_prediction]';
    % D) Update solveru (bez vytváření nového objektu)
    solver.update_problem(meas_window, x0_new, P0_new, initial_guess);

    % E) Optimalizace
    solver.opt();

    % F) Uložení výsledků
    current_states = solver.states';
    smooth_estimate{:,indices(1)} = current_states';

    % Ukládáme filtrovanou hodnotu (poslední bod v okně)
    estimated_states_history(:, k) = current_states(:, end);

    % Kovariance
    P_full = solver.compute_covariance();

    % Uložíme varianci posledního bodu pro vizualizaci
    idx_last = (WINDOW_SIZE-1)*model.nx + (1:model.nx);
    estimated_variance_history{k} = P_full(idx_last, idx_last);

end
toc
fprintf('Window hotovo.\n');

% vypocet různě vyhlazenych odhadů
max_lag = WINDOW_SIZE - 1;
base_row = WINDOW_SIZE;
for lag = 0 : max_lag
    clear est_k;
    for j = 1 : size(smooth_estimate, 2)
        row_idx = base_row - lag;
        est_k(:, j) = smooth_estimate{j}(row_idx, :);
    end
    RMSE_smooth = sqrt(mean((est_k - x_true_all(:,WINDOW_SIZE-lag:end-lag)).^2, 2));
    % Výpis výsledku
    RMSE_smooth_hist(:,lag+1) = RMSE_smooth;
end


%% KF
myFilter = TrajectoryFilters(model, Q, R, x0, P0);
tic
[x_est_KF, P_hist_KF] = myFilter.runKF(meas_all);
toc
for i = 1:size(P_hist_KF,3)
    P_hist_KF_cell{i} = P_hist_KF(:,:,i);
end

%% RTSS
tic
[x_est_RTSS, P_hist_RTSS] = myFilter.runRTSS(x_est_KF, P_hist_KF);
toc
for i = 1:size(P_hist_RTSS,3)
    P_hist_RTSS_cell{i} = P_hist_RTSS(:,:,i);
end

%% FG GTSAM

estimated_states_FG_GTSAM = vysledky_linear(:,7:9);

load("full_covariance.csv")
for i = 1:N
    P_hist_FG_GTSAM_cell{i} = reshape(full_covariance(i,2:10),[3,3])';
end

%% FG MATLAB
estimated_states_FG = zeros(model.nx, N);
estimated_variances = cell(1, N);
solver = FactorGraphSolver_LS(model, meas_all, R, Q, x0, P0);
tic
solver.opt();
toc
estimated_states_FG = solver.states' ;

P_full = solver.compute_covariance();
for t = 1:N
    idx_s = (t-1)*model.nx + 1;
    idx_e = t*model.nx;
    P_t = full(P_full(idx_s:idx_e, idx_s:idx_e));
    estimated_variances{t} = P_t;
end

%% VYHODNOCENÍ
RMSE_KF = sqrt(mean((x_est_KF - x_true_all).^2, 2));
RMSE_FG = sqrt(mean((estimated_states_FG - x_true_all).^2, 2));
RMSE_RTSS = sqrt(mean((x_est_RTSS - x_true_all).^2, 2));
RMSE_FG_GTSAM = sqrt(mean((estimated_states_FG_GTSAM - x_true_all').^2, 1))';
fprintf('Výsledné RMSE KF:\n');
disp(RMSE_KF)
fprintf('Výsledné RMSE FG-MATLAB:\n');
disp(RMSE_FG)
fprintf('Výsledné RMSE FG-GTSAM:\n');
disp(RMSE_FG_GTSAM)
fprintf('Výsledné RMSE RTSS:\n');
disp(RMSE_RTSS)

fprintf('KF:');
ANEES(x_est_KF,x_true_all,P_hist_KF_cell,model);

fprintf('FG-MATLAB:');
ANEES(estimated_states_FG,x_true_all,estimated_variances,model);

fprintf('FG-GTSAM:');
ANEES(estimated_states_FG_GTSAM',x_true_all,P_hist_FG_GTSAM_cell,model);

fprintf('RTSS:');
ANEES(x_est_RTSS,x_true_all,P_hist_RTSS_cell,model);




%% ========================================================================
% GRAF 1: Vliv zpoždění vyhlazovacího okna na RMSE
% =========================================================================
fig1 = figure('Color', 'w', 'Position',[200, 200, 1200, 600]);
l_values = 0:(WINDOW_SIZE - 1);
p = plot(l_values, RMSE_smooth_hist, 'LineWidth', 2, 'MarkerSize', 6);

if length(p) >= 3
    p(1).Marker = 'o'; 
    p(2).Marker = 's'; 
    p(3).Marker = '^'; 
end
grid on;
ax = gca;
ax.GridAlpha = 0.3; 
ax.FontSize = 15;   

xticks(l_values);
xlim([0, WINDOW_SIZE - 1]);

xlabel('Vyhlazení l [kroky]', 'Interpreter', 'tex', 'FontSize', 17);
ylabel('RMSE X_{k|k+l}', 'Interpreter', 'tex', 'FontSize', 17);
title('Vliv zpoždění vyhlazovacího okna na přesnost odhadu', 'Interpreter', 'tex', 'FontSize', 15);
lgd = legend('Poloha x', 'Rychlost v_x', 'Bias b', 'Interpreter', 'tex', 'Location', 'northeast');
lgd.FontSize = 15;


%% ========================================================================
% GRAF 2: Porovnání odhadů (RTSS, FG, KF) - S oddělenými markery
% =========================================================================
fig2 = figure('Name', 'Porovnání FG a RTSS', 'Color', 'w', 'Position',[200, 200, 1000, 800]);
t = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Definice kroků pro markery, aby se nepřekrývaly (každý 6. bod, ale posunutý)
idx_RTSS  = 1:6:100; % Markery na pozicích 1, 7, 13...
idx_GTSAM = 3:6:100; % Markery na pozicích 3, 9, 15...
idx_MATLAB= 5:6:100; % Markery na pozicích 5, 11, 17...

% --- Poloha ---
nexttile;
hold on; grid on;
plot(x_true_all(1,:), 'k-', 'LineWidth', 1.5); 
plot(x_est_RTSS(1,:), 'm-s', 'LineWidth', 1.2, 'MarkerSize', 5, 'MarkerIndices', idx_RTSS);
plot(estimated_states_FG_GTSAM(:,1), 'r-o', 'LineWidth', 1.2, 'MarkerSize', 4, 'MarkerIndices', idx_GTSAM);
plot(estimated_states_FG(1,:), 'b-d', 'LineWidth', 1.2, 'MarkerSize', 4, 'MarkerIndices', idx_MATLAB);
plot(x_est_KF(1,:), 'g:', 'LineWidth', 1.5); 
xlim([0 100]);
ylabel('Poloha x [m]', 'Interpreter', 'tex', 'FontSize', 17);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 14;

% --- Rychlost ---
nexttile;
hold on; grid on;
plot(x_true_all(2,:), 'k-', 'LineWidth', 1.5);
plot(x_est_RTSS(2,:), 'm-s', 'LineWidth', 1.2, 'MarkerSize', 5, 'MarkerIndices', idx_RTSS);
plot(estimated_states_FG_GTSAM(:,2), 'r-o', 'LineWidth', 1.2, 'MarkerSize', 4, 'MarkerIndices', idx_GTSAM);
plot(estimated_states_FG(2,:), 'b-d', 'LineWidth', 1.2, 'MarkerSize', 4, 'MarkerIndices', idx_MATLAB);
plot(x_est_KF(2,:), 'g:', 'LineWidth', 1.5); 
xlim([0 100]);
ylabel('Rychlost v [m/s]', 'Interpreter', 'tex', 'FontSize', 17);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 14;

% --- Bias ---
nexttile;
hold on; grid on;
h1 = plot(x_true_all(3,:), 'k-', 'LineWidth', 1.5);
h2 = plot(x_est_RTSS(3,:), 'm-s', 'LineWidth', 1.2, 'MarkerSize', 5, 'MarkerIndices', idx_RTSS, 'MarkerFaceColor', 'm');
h3 = plot(estimated_states_FG_GTSAM(:,3), 'r-o', 'LineWidth', 1.2, 'MarkerSize', 4, 'MarkerIndices', idx_GTSAM, 'MarkerFaceColor', 'r');
h4 = plot(estimated_states_FG(3,:), 'b-d', 'LineWidth', 1.2, 'MarkerSize', 4, 'MarkerIndices', idx_MATLAB, 'MarkerFaceColor', 'b');
h5 = plot(x_est_KF(3,:), 'g:', 'LineWidth', 1.5); 
xlim([0 100]);
ylabel('Bias b [m]', 'Interpreter', 'tex', 'FontSize', 17);
xlabel('Časový krok k', 'Interpreter', 'tex', 'FontSize', 17); 
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 14;

% --- Hlavní nadpis ---
title(t, '\bf Odhad stavů pomocí různých metod', 'Interpreter', 'tex', 'FontSize', 16);

% --- SPOLEČNÁ LEGENDA ---
lgd = legend([h1, h2, h3, h4, h5], 'Referenční trajektorie', 'RTSS (čtverce)', 'GTSAM FG (kolečka)', 'MATLAB FG (kosočtverce)', 'KF', ...
    'Interpreter', 'tex', 'NumColumns', 3);
lgd.FontSize = 12;
lgd.Layout.Tile = 'south';


%% ========================================================================
% GRAF 3: Odhad stavů s vizualizací nejistoty (3-sigma)
% =========================================================================
fig3 = figure('Name', 'Vizualizace nejistoty', 'Color', 'w', 'Position',[200, 200, 1000, 800]);
t = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
time_steps = 1:N;
state_names_tex = {"Poloha x [m]", "Rychlost v [m/s]", "Bias b [m]"};
sigma_bounds = zeros(N, 3);
for i = 1:N
    if ~isempty(estimated_variances{i})
        P_i_diag = diag(estimated_variances{i});
        sigma_bounds(i, :) = 3 * sqrt(max(P_i_diag, 1e-12))';
    end
end
upper = estimated_states_FG(1:3, :) + sigma_bounds'; 
lower = estimated_states_FG(1:3, :) - sigma_bounds';
for i = 1:3
    nexttile; 
    hold on; grid on;
    fill_x = [time_steps, fliplr(time_steps)];
    fill_y =[lower(i, :), fliplr(upper(i, :))];
    h_fill = fill(fill_x, fill_y, [0.8 0.9 0.8], 'FaceAlpha', 0.9, 'EdgeColor', 'none'); 
    h_true = plot(time_steps, x_true_all(i, :), 'k-', 'LineWidth', 1.5); 
    h_est  = plot(time_steps, estimated_states_FG(i, :), 'b--', 'LineWidth', 1.5);
    xlim([1 10]);
    ylabel(state_names_tex{i}, 'Interpreter', 'tex', 'FontSize', 17);
    ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 14;
    if i == 3
        xlabel('Časový krok k', 'Interpreter', 'tex', 'FontSize', 15);
    end
end
title(t, '\bf Odhad stavů s vizualizací nejistoty (3\sigma) - detail', 'Interpreter', 'tex', 'FontSize', 16);

% 5) SPOLEČNÁ LEGENDA DOLE
% Využije odkazy z posledního průběhu cyklu (h_fill, h_true, h_est)
lgd = legend([h_fill, h_true, h_est], '3\sigma interval', 'Referenční trajektorie', 'Odhad FG (MATLAB)', ...
    'Interpreter', 'tex', 'NumColumns', 3);
lgd.FontSize = 12;
lgd.Layout.Tile = 'south'; % Umístí legendu mimo grafy, na spodní okraj okna

exportgraphics(fig1, 'obrazky/LS/graf_RMSE_zpozdeni.pdf', 'ContentType', 'vector');
exportgraphics(fig2, 'obrazky/LS/graf_porovnani_metod.pdf', 'ContentType', 'vector');
exportgraphics(fig3, 'obrazky/LS/graf_nejistota_detail.pdf', 'ContentType', 'vector');