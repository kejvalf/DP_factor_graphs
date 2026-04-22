%% ========================================================================
% Spouštěcí skript pro FG estimaci - SLIDING WINDOW
% =========================================================================
%clc; clear; 
%close all;
rng(94);  

% ========================================================================
%  DEFINICE MODELU
% =========================================================================
model.nx = 4; % [x, y, vx, vy]
model.nz = 2; % [z_baro, pseudo_vx, pseudo_vy]
model.dt = 1.0;

load("data.mat")
xv = souradniceX(1,:);
yv = souradniceY(:,1);
[map_m.x,map_m.y] = meshgrid(xv,yv);
map_m.z = souradniceZ;
% Gradienty mapy
[Gx, Gy] = gradient(map_m.z, mean(diff(xv)), mean(diff(yv)));
H_map    = griddedInterpolant({yv,xv}, map_m.z, 'linear','nearest');
dHdx_map = griddedInterpolant({yv,xv}, Gx, 'linear','nearest');
dHdy_map = griddedInterpolant({yv,xv}, Gy, 'linear','nearest');


% Funkce měření h(x) - původní
% model.H = @(x) [H_map(x(2), x(1));
%                 (x(3)./sqrt(x(3).^2+x(4).^2)).*x(3)-(x(4)./sqrt(x(3).^2+x(4).^2)).*x(4);
%                 (x(4)./sqrt(x(3).^2+x(4).^2)).*x(3)+(x(3)./sqrt(x(3).^2+x(4).^2)).*x(4)];
% Funkce měření h(x) - aktualizovaná
model.H = @(x) [H_map(x(2), x(1));
                sqrt(x(3)^2 + x(4)^2)];
% Funkce měření h(x) - linearni
% model.H = @(x) [H_map(x(2), x(1));
%     x(3);
%     x(4)];


% Jacobián měření H(x) - původní
% model.dHdX = @(x) [dHdx_map(x(2), x(1)), dHdy_map(x(2), x(1)), 0, 0;
%                      0,                  0,                    (x(3)^3 + 3*x(3)*x(4)^2)/sqrt((x(3)^2 + x(4)^2)^3), -(3*x(3)^2*x(4) + x(4)^3)/sqrt((x(3)^2 + x(4)^2)^3);
%                      0,                  0,                    2*x(4)^3/sqrt((x(3)^2 + x(4)^2)^3),                 2*x(3)^3/sqrt((x(3)^2 + x(4)^2)^3)];
% Jacobián měření H(x) - aktualizovaný
model.dHdX = @(x) [dHdx_map(x(2), x(1)), dHdy_map(x(2), x(1)), 0                          ,0;
                     0                 , 0                   ,x(3) / (sqrt(x(3)^2 + x(4)^2)), x(4) / (sqrt(x(3)^2 + x(4)^2))];
% Jacobián měření H(x) - linearni
% model.dHdX = @(x) [dHdx_map(x(2), x(1)), dHdy_map(x(2), x(1)), 0, 0;
%     0,                  0,                    1, 0;
%     0,                  0,                    0, 1];


% Dynamika (pohyb konstantní rychlostí)
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
stdV = 0.1;
%=============================================================
% R = diag([5^2, stdV^2, stdV^2]);
R = diag([0.5^2, stdV^2]);
%=============================================================

%% PŘÍPRAVA DAT (GLOBÁLNĚ)
% Všechna data připravíme najednou před cyklem, aby byla konzistentní
data_range = 1:400;%size(souradniceGNSS,2); % Celý dataset nebo např. 800:900,  max: 1276
souradniceGNSS = souradniceGNSS(:, data_range);
hB = hB(:, data_range);

poloha = souradniceGNSS(1:2, :);
% Dopočet rychlosti pro Ground Truth
rychlost = [diff(poloha, 1, 2), [0;0]];
rychlost(:, end) = rychlost(:, end-1);

x_true_all = [poloha; rychlost];
N_total = size(x_true_all, 2);

% Generování měření pro celou trajektorii
meas_all = zeros(model.nz, N_total);
for k = 1:N_total
    xk = x_true_all(:, k);
   
    %===================================================================================================================
    % Teoretické měření + šum - původní
    % v_norm = sqrt(xk(3)^2 + xk(4)^2);
    % if v_norm == 0, v_norm = eps; end
    % meas_clean = [hB(k);
    %               (xk(3)/v_norm)*xk(3) - (xk(4)/v_norm)*xk(4);
    %               (xk(4)/v_norm)*xk(3) + (xk(3)/v_norm)*xk(4)];

    % Teoretické měření + šum - aktualizované
    meas_clean = [H_map(xk(2),xk(1)); %hB(k)
                  sqrt(xk(3)^2+xk(4)^2)];

    % Linearni měření + šum
    % meas_clean = [H_map(xk(2),xk(1));
    %     xk(3);
    %     xk(4)];
    %===================================================================================================================

    % Přidání šumu
    meas_all(:, k) = meas_clean + sqrt(diag(R)) .* randn(model.nz,1);
    %meas_all(1, k) = hB(k); % realne data
end


% Inicializace Priori (pro úplně první bod)
P0 = diag([10, 10, 0.5, 0.5]).^2; 
x0 = x_true_all(:,1) + sqrt(P0) * randn(4,1);

%% --------------------------------------------
% POUŽITÍ TŘÍDY TrajectoryFilter na EKF a UKF
% --------------------------------------------
    myFilter = TrajectoryFilters(model, Q, R, x0, P0);
    
    % Spuštění EKF
    fprintf('Spouštím EKF...\n');
    tic;
    [x_est_EKF, P_hist_EKF] = myFilter.runEKF(meas_all);
    t_ekf = toc;
    fprintf('EKF hotovo za %.4f s\n', t_ekf);
    RMSE_EKF = sqrt(mean((x_est_EKF - x_true_all).^2, 2));
    fprintf('EKF RMSE: \n');
    disp(RMSE_EKF);
    
    % Spuštění UKF
    fprintf('Spouštím UKF...\n');
    tic;
    [x_est_UKF, P_hist_UKF] = myFilter.runUKF(meas_all);
    t_ukf = toc;
    fprintf('UKF hotovo za %.4f s\n', t_ukf);
    RMSE_UKF = sqrt(mean((x_est_UKF - x_true_all).^2, 2));
    fprintf('UKF RMSE: \n');
    disp(RMSE_UKF);

%% ========================================================================
% Vykreslení výsledků: EKF a UKF
% =========================================================================

fig_ekf_ukf = figure('Name', 'Výsledky EKF a UKF', 'Color', 'w', 'Position',[150, 150, 900, 700]);
t_layout1 = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Zjištění počtu bodů pro rozložení značek (markerů)
N_pts = size(x_true_all, 2);
N_vel = N_pts - 1; % Rychlosti vykresluješ o 1 bod kratší
space = 15; % Rozestup mezi značkami

% Fázově posunuté indexy
m_idx_true = 1 : space : N_pts;
m_idx_ekf  = 6 : space : N_pts;
m_idx_ukf  = 11 : space : N_pts;

m_idx_true_v = 1 : space : N_vel;
m_idx_ekf_v  = 6 : space : N_vel;
m_idx_ukf_v  = 11 : space : N_vel;

% --- 1. Poloha X ---
nexttile;
hold on; grid on;
h1 = plot(x_true_all(1,:), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
h2 = plot(x_est_EKF(1,:), 'b-', 'LineWidth', 1.5, ...
    'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_ekf, 'MarkerFaceColor', 'w');
h3 = plot(x_est_UKF(1,:), 'r-', 'LineWidth', 1.5, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_ukf, 'MarkerFaceColor', 'w');
ylabel('Poloha {\it x} [m]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
xticklabels({}); % Skryje osu X

% --- 2. Poloha Y ---
nexttile;
hold on; grid on;
plot(x_true_all(2,:), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
plot(x_est_EKF(2,:), 'b-', 'LineWidth', 1.5, ...
    'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_ekf, 'MarkerFaceColor', 'w');
plot(x_est_UKF(2,:), 'r-', 'LineWidth', 1.5, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_ukf, 'MarkerFaceColor', 'w');
ylabel('Poloha {\it y} [m]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
xticklabels({}); % Skryje osu X

% --- 3. Rychlost X ---
nexttile;
hold on; grid on;
plot(x_true_all(3,1:end-1), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true_v, 'MarkerFaceColor', 'w');
plot(x_est_EKF(3,1:end-1), 'b-', 'LineWidth', 1.5, ...
    'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_ekf_v, 'MarkerFaceColor', 'w');
plot(x_est_UKF(3,1:end-1), 'r-', 'LineWidth', 1.5, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_ukf_v, 'MarkerFaceColor', 'w');
xlabel('Časový krok {\it k}', 'Interpreter', 'tex', 'FontSize', 14);
ylabel('Rychlost {\it v}_x [m/s]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;

% --- 4. Rychlost Y ---
nexttile;
hold on; grid on;
plot(x_true_all(4,1:end-1), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true_v, 'MarkerFaceColor', 'w');
plot(x_est_EKF(4,1:end-1), 'b-', 'LineWidth', 1.5, ...
    'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_ekf_v, 'MarkerFaceColor', 'w');
plot(x_est_UKF(4,1:end-1), 'r-', 'LineWidth', 1.5, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_ukf_v, 'MarkerFaceColor', 'w');
xlabel('Časový krok {\it k}', 'Interpreter', 'tex', 'FontSize', 14);
ylabel('Rychlost {\it v}_y [m/s]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;

% --- Nadpis a sdílená legenda ---
title(t_layout1, '\bf Porovnání odhadů EKF a UKF pro vybraný běh MC', 'Interpreter', 'tex', 'FontSize', 16);
lgd1 = legend([h1, h2, h3], 'Referenční trajektorie', 'EKF', 'UKF', ...
    'Interpreter', 'tex', 'Orientation', 'horizontal', 'NumColumns', 3);
lgd1.FontSize = 12;
lgd1.Layout.Tile = 'south'; 

exportgraphics(fig_ekf_ukf, 'obrazky/TAN/EKF_UKF_cherryMC.pdf', 'ContentType', 'vector');


%% ========================================================================
%  3. NASTAVENÍ POSUVNÉHO OKNA
% =========================================================================
WINDOW_SIZE = 3;
 % Velikost okna : sum(RMSE)
 % 20: 377
 % 30: 419
 % 35: 376
 % 36: 380
 % 37: 250 ---------------
 % 38: 251
 % 39: 254
 % 40: 256
 % 50: 342 
 % 80: 425
 % 100: 443
 % 120: 4925
 % 150: 387
 % 170: 385
 % 180: 358
 % 200: 7702
fprintf('Spouštím Sliding Window Estimaci (okno = %d)...\n', WINDOW_SIZE);

% Pole pro výsledky
estimated_states_history = zeros(model.nx, N_total);
estimated_variance_history = cell(1, N_total);

% --- INICIALIZACE PRVNÍHO OKNA ---
meas_window = meas_all(:, 1:WINDOW_SIZE);

% Vytvoříme objekt solveru (jednou)
solver = FactorGraphSolver_window(model, meas_window, R, Q, x0, P0);
solver.opt();

% Uložení výsledků z prvního okna
current_states = solver.states'; % [nx x WINDOW_SIZE]
estimated_states_history(:, 1:WINDOW_SIZE) = current_states;
smooth_estimate{:,1} = current_states';
% struktura smooth odhadu:
% cell {1}  cell{2}     ...
% x_1/10    x_2/11
% x_2/10    x_3/11
% x_3/10    x_4/11
% ...       ...
% x_10/10   x_11/11

P_full = solver.compute_covariance();
for t = 1:WINDOW_SIZE
    idx = (t-1)*model.nx + (1:model.nx);
    estimated_variance_history{t} = P_full(idx, idx);
end

% ========================================================================
%  4. HLAVNÍ SMYČKA (Sliding Window)
% =========================================================================
tic;
for k = WINDOW_SIZE + 1 : N_total
    
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
    
    initial_guess = [prev_trajectory_shifted, new_prediction]'; % Transponovat na [N x nx]
    
    % D) Update solveru (bez vytváření nového objektu)
    solver.update_problem(meas_window, x0_new, P0_new, initial_guess);
    
    % E) Optimalizace
    solver.opt();
    
    % F) Uložení výsledků
    current_states = solver.states';
    smooth_estimate{:,indices(1)} = current_states';

    % Ukládáme FILTROVANOU hodnotu (poslední bod v okně)
    %estimated_states_history(:, k) = current_states(:, end);
    % Ukladame VYHLAZENOU trajektorii přes WINOW_SIZE stavů
    estimated_states_history(:, indices) = current_states(:, 1:end);
    
    % Kovariance
    P_full = solver.compute_covariance();
    
    % Uložíme varianci posledního bodu pro vizualizaci
    idx_last = (WINDOW_SIZE-1)*model.nx + (1:model.nx);
    estimated_variance_history{k} = P_full(idx_last, idx_last);
end
toc
fprintf('Hotovo.\n');

% ========================================================================
%  5. VIZUALIZACE A VYHODNOCENÍ
% =========================================================================
% Oříznutí referenčních dat na délku simulace
x_true_all = x_true_all(:, 1:N_total);
estimated_states_history  = estimated_states_history(:, 1:N_total);

% struktura smooth_estimate:
% cell {1}  cell{2}  cell{3}   cell{4}   cell{5}   cell{6}   cell{7}   cell{8}   cell{8}   cell{9}  ...   cell{N}    
% x_1|10    x_2|11   x_3|12    x_4|13    x_5|14    x_6|15    x_7|16    x_8|17    x_9|18    x_10|19                              
% x_2|10    x_3|11   x_4|12    x_5|13    x_6|14    x_7|15    x_8|16    x_9|17    x_10|18   x_11|19                      
% x_3|10    x_4|11   x_5|12    x_6|13    x_7|14    x_8|15    x_9|16    x_10|17   x_11|18   x_12|19             
% x_4|10    x_5|11   x_6|12    x_7|13    x_8|14    x_9|15    x_10|16   x_11|17   x_12|18   x_13|19    
% x_5|10    x_6|11   x_7|12    x_8|13    x_9|14    x_10|15   x_11|16   x_12|17   x_13|18   x_14|19
% x_6|10    x_7|11   x_8|12    x_9|13    x_10|14   x_11|15   x_12|16   x_13|17   x_14|18   x_15|19          
% x_7|10    x_8|11   x_9|12    x_10|13   x_11|14   x_12|15   x_13|16   x_14|17   x_15|18   x_16|19
% x_8|10    x_9|11   x_10|12   x_11|13   x_12|14   x_13|15   x_14|16   x_15|17   x_16|18   x_17|19
% x_9|10    x_10|11  x_11|12   x_12|13   x_13|14   x_14|15   x_15|16   x_16|17   x_17|18   x_18|19
% x_10|10   x_11|11  x_12|12   x_13|13   x_14|14   x_15|15   x_16|16   x_17|17   x_18|18   x_19|19


% max_lag = WINDOW_SIZE - 1; 
% base_row = WINDOW_SIZE ;
% for k = 0 : max_lag
%     clear est_k; 
%     for j = (k + 1) : size(smooth_estimate, 2)
%         row_idx = base_row - k; 
%         est_k(:, j) = smooth_estimate{j}(row_idx, :);
%     end
%     est_slice = est_k(:, (k + 1):end);
%     true_slice = x_true_all(:, WINDOW_SIZE:(end - k));
%     RMSE_smooth = sqrt(mean((est_slice - true_slice).^2, 2));
%     % Výpis výsledku
%     RMSE_smooth_hist(:,k+1) = RMSE_smooth;
%     fprintf('\nFG vyhlazený RMSE x_k|k+%d:\n', k);
%     disp(RMSE_smooth);
% end
% 
% figure
% plot(linspace(0,WINDOW_SIZE-1,WINDOW_SIZE),RMSE_smooth_hist)
% grid on
% ylabel("RMSE x(k) - x_(k|k+l)")
% xlabel("l")
% legend("x","y","x_v","y_v")
% title("Vliv vyhlazení na RMSE")


% Výpočet RMSE filtračního odhadu
RMSE = sqrt(mean((estimated_states_history(:,WINDOW_SIZE:end) - x_true_all(:,WINDOW_SIZE:end)).^2, 2));
fprintf('\nFG Vyhlazený RMSE:\n');
disp(RMSE);


%% ========================================================================
% Vykreslení výsledků: Faktorový graf (Klouzavé okno)
% =========================================================================

fig_fg = figure('Name', 'Výsledky FG (MATLAB)', 'Color', 'w', 'Position',[200, 200, 900, 700]);
t_layout2 = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

m_idx_fg   = 8 : space : N_pts;
m_idx_fg_v = 15 : space : N_vel;

% --- 1. Poloha X ---
nexttile;
hold on; grid on;
h1_fg = plot(x_true_all(1,:), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
h2_fg = plot(estimated_states_history(1,:), 'b-', 'LineWidth', 1.5, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_fg, 'MarkerFaceColor', 'w');
ylabel('Poloha {\it x} [m]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
xticklabels({}); % Skryje osu X

% --- 2. Poloha Y ---
nexttile;
hold on; grid on;
plot(x_true_all(2,:), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
plot(estimated_states_history(2,:), 'b-', 'LineWidth', 1.5, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_fg, 'MarkerFaceColor', 'w');
ylabel('Poloha {\it y} [m]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
xticklabels({}); % Skryje osu X

% --- 3. Rychlost X ---
nexttile;
hold on; grid on;
plot(x_true_all(3,1:end-1), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true_v, 'MarkerFaceColor', 'w');
plot(estimated_states_history(3,1:end-1), 'b-', 'LineWidth', 1.5, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_fg_v, 'MarkerFaceColor', 'w');
xlabel('Časový krok {\it k}', 'Interpreter', 'tex', 'FontSize', 14);
ylabel('Rychlost {\it v}_x [m/s]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;

% --- 4. Rychlost Y ---
nexttile;
hold on; grid on;
plot(x_true_all(4,1:end-1), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true_v, 'MarkerFaceColor', 'w');
plot(estimated_states_history(4,1:end-1), 'b-', 'LineWidth', 1.5, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_fg_v, 'MarkerFaceColor', 'w');
xlabel('Časový krok {\it k}', 'Interpreter', 'tex', 'FontSize', 14);
ylabel('Rychlost {\it v}_y [m/s]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;

% --- Nadpis a sdílená legenda ---
title(t_layout2, '\bf Odhad pomocí FG (MATLAB, klouzavé okno delky 3) pro vybraný běh MC', 'Interpreter', 'tex', 'FontSize', 16);
lgd2 = legend([h1_fg, h2_fg], 'Referenční trajektorie', 'FG (MATLAB)', ...
    'Interpreter', 'tex', 'Orientation', 'horizontal', 'NumColumns', 2);
lgd2.FontSize = 12;
lgd2.Layout.Tile = 'south';

exportgraphics(fig_fg, 'obrazky/TAN/FG_cherryMC.pdf', 'ContentType', 'vector');




% Volání vykreslovací funkce
%plot_results(N_total, x_true_all, estimated_states_history, estimated_variance_history, hB, map_m);


% -------------------------------------------------------------------------
% Pomocná funkce pro vykreslení
% -------------------------------------------------------------------------
function plot_results(N, true_states, estimated_states, estimated_variances, hB, map_m)
    time_steps = 1:N;
    
    % 1. Vykreslení stavů s intervalem spolehlivosti
    figure('Name', 'Výsledky FG Sliding Window Estimace', 'Color', 'w');
    state_names = ["Pozice X [m]", "Pozice Y [m]", "Rychlost X [m/s]", "Rychlost Y [m/s]"];
    
    sigma_bounds = zeros(N, 4);
    for i = 1:N
        if ~isempty(estimated_variances{i})
            P_diag = diag(estimated_variances{i});
            sigma_bounds(i, :) = 3 * sqrt(max(P_diag, 1e-12))';
        end
    end
    
    upper = estimated_states + sigma_bounds';
    lower = estimated_states - sigma_bounds';

    for i = 1:4
        subplot(2, 2, i);
        fill_x = [time_steps, fliplr(time_steps)];
        fill_y = [lower(i, :), fliplr(upper(i, :))];
        
        fill(fill_x, fill_y, [0.8 0.9 0.8], 'FaceAlpha', 0.6, 'EdgeColor', 'none'); hold on;
        plot(time_steps, true_states(i, :), 'r-', 'LineWidth', 1.5);
        plot(time_steps, estimated_states(i, :), 'b--', 'LineWidth', 1.2);
        
        title(state_names(i)); grid on; axis tight;
        if i==1, legend('3\sigma Conf', 'True', 'Estimate', 'Location','best'); end
    end
    
    % 2. 3D Mapa
    figure('Name', 'Trajektorie FG na mapě', 'Color', 'w');
    surf(map_m.x, map_m.y, map_m.z, 'EdgeColor', 'none', 'FaceAlpha', 1); 
    colormap parula; shading interp; hold on;
    
    
    plot3(true_states(1,:), true_states(2,:), hB(1:N) , 'r', 'LineWidth', 2);
    plot3(estimated_states(1,:), estimated_states(2,:), hB(1:N) , 'b--', 'LineWidth', 1.5);
    
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Výška [m]');
    legend('Terén', 'True', 'Estimate');
    view(3); grid on;
end