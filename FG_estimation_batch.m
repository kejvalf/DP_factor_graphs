%% ========================================================================
% Spouštěcí skript pro FG estimaci
% =========================================================================
clc; clear;
close all;
rng(12);

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
data_range = 1:200;%size(souradniceGNSS,2); % Celý dataset nebo např. 800:900
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
    % meas_clean = [H_map(xk(2),xk(1)); %hB(k)
    %               sqrt(xk(3)^2+xk(4)^2)];

    % Linearni měření + šum
    meas_clean = [H_map(xk(2),xk(1));
        xk(3);
        xk(4)];
    %===================================================================================================================

    % Přidání šumu
    %meas_all(:, k) = meas_clean + sqrt(diag(R)) .* randn(model.nz,1);
    %meas_all(1, k) = hB(k); % realne data
end


%% přepsání vstupních měření daty z c++ (GTSAM)
vysledky_gtsam = load("vysledky_gtsam_nelin_mer.csv");
% vysledky_gtsam = load("vysledky_gtsam_lin_mer.csv");
meas_all = vysledky_gtsam(:,8:9)'; % NELIN
% meas_all = vysledky_gtsam(:,8:10)'; % LIN

%% NASTAVENÍ BATCH ESTIMACE
BATCH_SIZE = size(data_range,2); %  max: 1276
num_batches = ceil(N_total / BATCH_SIZE);

% Pole pro ukládání výsledků
estimated_states = zeros(model.nx, N_total);
estimated_variances = cell(1, N_total);

% Inicializace pro PRVNÍ batch
P0 = diag([1, 1, 0.01, 0.01]).^2;  % přesně znám poč. podm.
x0 = x_true_all(:,1) + sqrt(P0) * randn(4,1);
meas_init = meas_all(:, 1:min(BATCH_SIZE, N_total));
solver = FactorGraphSolver(model, meas_init, R, Q, x0, P0);
current_idx = 1;

%% HLAVNÍ SMYČKA FG
fprintf('Spouštím FG-MATLAB ');
fprintf("Čas optimalizace: \n")
for k = 1:num_batches
    % 1. Výběr dat pro aktuální batch
    idx_start = (k-1) * BATCH_SIZE + 1;
    idx_end   = min(k * BATCH_SIZE, N_total);
    indices   = idx_start:idx_end;
    meas_batch = meas_all(:, indices);
    % 2. Update Solveru (Recyklace objektu)
    % Předáme nová měření a Prior (x0, P0) pro začátek tohoto batche
    solver.update_problem(meas_batch, x0, P0);
    % 3. Optimalizace
    tic
    solver.opt();
    toc
    % 4. Získání výsledků a uložení
    est_state_batch = solver.states';
    n_samples = size(est_state_batch, 2);
    % Uložení stavů
    idx_range = current_idx : (current_idx + n_samples - 1);
    estimated_states(:, idx_range) = est_state_batch;

    % Výpočet a uložení kovariance
    P_full_batch = solver.compute_covariance();
    for t = 1:n_samples
        idx_s = (t-1)*model.nx + 1;
        idx_e = t*model.nx;
        P_t = full(P_full_batch(idx_s:idx_e, idx_s:idx_e));
        estimated_variances{current_idx + t - 1} = P_t;
    end

    % 5. PŘÍPRAVA PRO DALŠÍ BATCH (STITCHING / NAPOJENÍ)
    if k < num_batches
        % a) Vezmeme poslední odhad z aktuálního batche
        x_end = est_state_batch(:, end);
        P_end = estimated_variances{current_idx + n_samples - 1};
        % Predikce stavu: x_{k+1} = F(x_k)
        x0 = model.F(x_end);
        % Predikce kovariance: P_{k+1} = F * P_k * F' + Q
        F_jac = model.dFdX(x_end);
        P0 = F_jac * P_end * F_jac' + Q;
        % Udržení symetrie kovariance
        P0 = (P0 + P0') / 2;
    end

    current_idx = current_idx + n_samples;
end


% --------------------------------------------
%% POUŽITÍ TŘÍDY TrajectoryFilter na EKF a UKF
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

% Spouštím ERTS
fprintf('Spouštím ERTS...\n');
tic;
[x_est_ERTS, P_hist_ERTS] = myFilter.runRTSS(x_est_EKF, P_hist_EKF);
t_ekf = toc;
fprintf('ERTSS hotovo za %.4f s\n', t_ekf);
RMSE_ERTS = sqrt(mean((x_est_ERTS - x_true_all).^2, 2));
fprintf('ERTSS RMSE: \n');
disp(RMSE_ERTS);

% Spouštění IEKF
fprintf('Spouštím IEKF...\n');
tic;
[x_est_IEKF, P_hist_IEKF] = myFilter.runIEKF(meas_all,100,1e-6);
t_ekf = toc;
fprintf('IEKF hotovo za %.4f s\n', t_ekf);
RMSE_IEKF = sqrt(mean((x_est_IEKF - x_true_all).^2, 2));
fprintf('IEKF RMSE: \n');
disp(RMSE_IEKF);


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
% Vykreslení výsledků EKF a UKF (Plné čáry, posunuté markery pro čb tisk)
% =========================================================================

fig_ekf_ukf = figure('Name', 'Výsledky EKF a UKF', 'Color', 'w', 'Position',[150, 150, 900, 700]);

% Vytvoření mřížky 2x2
t_layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Zjištění počtu bodů pro rozložení značek
N_pts = size(x_true_all, 2);
N_vel = N_pts - 1; % Rychlosti mají o 1 bod méně

space = 50; % Rozestup mezi značkami na jedné čáře

% Fázově posunuté indexy pro POLOHU (aby se značky nestřílely přes sebe)
m_idx_true = 1 : space : N_pts;
m_idx_ekf  = 11 : space : N_pts;
m_idx_ukf  = 22 : space : N_pts;

% Fázově posunuté indexy pro RYCHLOST
m_idx_true_v = 1 : space : N_vel;
m_idx_ekf_v  = 11 : space : N_vel;
m_idx_ukf_v  = 22 : space : N_vel;

% --- 1. Poloha X ---
nexttile;
hold on; grid on;

% Uložíme handly pro legendu. Každá čára používá své posunuté indexy.
h1 = plot(x_true_all(1,:), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
h2 = plot(x_est_EKF(1,:), 'b-', 'LineWidth', 1.5, ...
    'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_ekf, 'MarkerFaceColor', 'w');
h3 = plot(x_est_UKF(1,:), 'r-', 'LineWidth', 1.5, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_ukf, 'MarkerFaceColor', 'w');

ylabel('Poloha {\it x} [m]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
xticklabels({}); % Skryje čísla na ose X pro horní graf

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
xticklabels({}); % Skryje čísla na ose X pro horní graf

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

% --- Hlavní nadpis a sdílená legenda ---
title(t_layout, '\bf Porovnání odhadů EKF a UKF', 'Interpreter', 'tex', 'FontSize', 16);

lgd = legend([h1, h2, h3], 'Referenční trajektorie', 'EKF', 'UKF', ...
    'Interpreter', 'tex', 'Orientation', 'horizontal');
lgd.FontSize = 12;
lgd.Layout.Tile = 'south'; % Umístí legendu krásně pod všechny 4 grafy

exportgraphics(fig_ekf_ukf, 'obrazky/TAN/EKF_UKF.pdf', 'ContentType', 'vector');


%% ========================================================================
% Vykreslení výsledků EKF, ERTSS a IEKF
% =========================================================================

fig_compare = figure('Name', 'Porovnání filtrů: EKF, ERTSS, IEKF', 'Color', 'w', 'Position',[150, 150, 900, 700]);

% Vytvoření mřížky 2x2
t_layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Zjištění počtu bodů
N_pts = size(x_true_all, 2);
N_vel = size(x_true_all, 2); % Předpokládám stejnou délku, případně upravte na N_pts-1

space = 60; % Větší rozestup, protože máme více čar

% Fázově posunuté indexy pro markery (aby se nepřekrývaly)
m_idx_true  = 1 : space : N_pts;
m_idx_ekf   = 15 : space : N_pts;
m_idx_ertss = 30 : space : N_pts;
m_idx_iekf  = 45 : space : N_pts;

% Barvy a styly
col_true  = [0 0 0]; % Černá
col_ekf   = [1 0 0]; % červena
col_ertss = [0 1 0]; % zelena
col_iekf  = [0 0 1]; % modra

% --- Pomocná funkce pro vykreslení všech 4 čar do subgrafu ---
% (Tento blok kódu vložíme do každého tile)

titles_y = {'Poloha {\it x} [m]', 'Poloha {\it y} [m]', ...
            'Rychlost {\it v}_x [m/s]', 'Rychlost {\it v}_y [m/s]'};
state_idx = [1, 2, 3, 4];

for i = 1:4
    nexttile;
    hold on; grid on;
    idx = state_idx(i);
    
    % 1. Reference
    h1 = plot(x_true_all(idx,:), 'Color', col_true, 'LineStyle', '-', 'LineWidth', 1.5, ...
        'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
    
    % 2. EKF
    h2 = plot(x_est_EKF(idx,:), 'Color', col_ekf, 'LineStyle', '-', 'LineWidth', 1.5, ...
        'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_ekf, 'MarkerFaceColor', 'w');
    
    % 3. ERTSS (Vyhlazovač)
    h3 = plot(x_est_ERTS(idx,:), 'Color', col_ertss, 'LineStyle', '--', 'LineWidth', 1.5, ...
        'Marker', 'd', 'MarkerSize', 5, 'MarkerIndices', m_idx_ertss, 'MarkerFaceColor', 'w');
    
    % 4. IEKF (Iterační)
    h4 = plot(x_est_IEKF(idx,:), 'Color', col_iekf, 'LineStyle', ':', 'LineWidth', 1.5, ...
        'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_iekf, 'MarkerFaceColor', 'w');

    ylabel(titles_y{i}, 'Interpreter', 'tex', 'FontSize', 13);
    ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 11;
    
    if i <= 2
        xticklabels({}); % Skryje čísla na ose X pro horní řadu
    else
        xlabel('Časový krok {\it k}', 'Interpreter', 'tex', 'FontSize', 13);
    end
end

% --- Hlavní nadpis a sdílená legenda ---
title(t_layout, '\bf Porovnání nelineárních filtrů a vyhlazovače (EKF vs. ERTSS vs. IEKF)', ...
    'Interpreter', 'tex', 'FontSize', 15);

lgd = legend([h1, h2, h3, h4], ...
    'Referenční trajektorie', ...
    'EKF (Dopředný)', ...
    'ERTSS (Vyhlazený EKF)', ...
    'IEKF (Iterační)', ...
    'Interpreter', 'tex', 'Orientation', 'horizontal');

lgd.FontSize = 12;
lgd.Layout.Tile = 'south'; 

% Export
if ~exist('obrazky/TAN', 'dir'), mkdir('obrazky/TAN'); end
exportgraphics(fig_compare, 'obrazky/TAN/EKF_IEKF_ERTSS_comp.pdf', 'ContentType', 'vector');

%% VYHODNOCENÍ
% --- 1. Výpočet RMSE ---
RMSE_FG = sqrt(mean((estimated_states - x_true_all).^2, 2));
fprintf('\nVýsledné RMSE MATLAB FG:\n X: %.2f m\n Y: %.2f m\n Vx: %.2f m/s\n Vy: %.2f m/s\n', RMSE_FG);

RMSE_FG_GTSAM = sqrt(mean((vysledky_gtsam(:,2:5)' - x_true_all).^2, 2));
fprintf('\nVýsledné RMSE GTSAM:\n X: %.2f m\n Y: %.2f m\n Vx: %.2f m/s\n Vy: %.2f m/s\n', RMSE_FG_GTSAM);

% --- 2. Výpočet ANEES ---

for i = 1:size(P_hist_EKF,3)
    P_hist_EKF_cell{i} = P_hist_EKF(:,:,i);
end
for i = 1:size(P_hist_ERTS,3)
    P_hist_ERTS_cell{i} = P_hist_ERTS(:,:,i);
end
for i = 1:size(P_hist_IEKF,3)
    P_hist_IEKF_cell{i} = P_hist_IEKF(:,:,i);
end
for i = 1:size(P_hist_UKF,3)
    P_hist_UKF_cell{i} = P_hist_UKF(:,:,i);
end

% Výpočet celkového průměru (ANEES)
fprintf("EKF ANEES: \n");
ANEES(x_est_EKF,x_true_all,P_hist_EKF_cell,model);
fprintf("ERTS ANEES: \n");
ANEES(x_est_ERTS,x_true_all,P_hist_ERTS_cell,model);
fprintf("IEKF ANEES: \n");
ANEES(x_est_IEKF,x_true_all,P_hist_IEKF_cell,model);
fprintf("UKF ANEES: \n");
ANEES(x_est_UKF,x_true_all,P_hist_UKF_cell,model);
fprintf("FG MATLAB ANEES: \n");
ANEES(estimated_states,x_true_all,estimated_variances,model);


%% ========================================================================
% Vykreslení výsledků: Porovnání MATLAB FG a GTSAM FG
% =========================================================================

fig_fg_comp = figure('Name', 'Porovnání MATLAB a GTSAM', 'Color', 'w', 'Position',[150, 150, 900, 700]);

% Vytvoření mřížky 2x2
t_layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Zjištění počtu bodů pro rozložení značek (markerů)
N_pts = size(estimated_states, 2);

space = 50; % Rozestup mezi značkami na jedné čáře

% Fázově posunuté indexy (aby se značky nepřekrývaly)
m_idx_true   = 1 : space : N_pts;
m_idx_matlab = 11 : space : N_pts;
m_idx_gtsam  = 22 : space : N_pts;

% --- 1. Poloha X ---
nexttile;
hold on; grid on;

% Uložíme handly pro legendu
h1 = plot(x_true_all(1,:), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
h2 = plot(estimated_states(1,:), 'b-', 'LineWidth', 1.5, ...
    'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_matlab, 'MarkerFaceColor', 'w');
h3 = plot(x_est_EKF(1,:), 'r-', 'LineWidth', 1.2, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_gtsam, 'MarkerFaceColor', 'w');

ylabel('Poloha {\it x} [m]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
xticklabels({}); % Skryje čísla na ose X pro horní graf

% --- 2. Poloha Y ---
nexttile;
hold on; grid on;
plot(x_true_all(2,:), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
plot(estimated_states(2,:), 'b-', 'LineWidth', 1.5, ...
    'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_matlab, 'MarkerFaceColor', 'w');
plot(x_est_EKF(2,:), 'r-', 'LineWidth', 1.2, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_gtsam, 'MarkerFaceColor', 'w');

ylabel('Poloha {\it y} [m]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
xticklabels({}); % Skryje čísla na ose X pro horní graf

% --- 3. Rychlost X ---
nexttile;
hold on; grid on;
plot(x_true_all(3,:), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
plot(estimated_states(3,:), 'b-', 'LineWidth', 1.5, ...
    'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_matlab, 'MarkerFaceColor', 'w');
plot(x_est_EKF(3,:), 'r-', 'LineWidth', 1.2, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_gtsam, 'MarkerFaceColor', 'w');

xlabel('Časový krok {\it k}', 'Interpreter', 'tex', 'FontSize', 14);
ylabel('Rychlost {\it v}_x [m/s]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;

% --- 4. Rychlost Y ---
nexttile;
hold on; grid on;
plot(x_true_all(4,:), 'k-', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
plot(estimated_states(4,:), 'b-', 'LineWidth', 1.5, ...
    'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_matlab, 'MarkerFaceColor', 'w');
plot(x_est_EKF(4,:), 'r-', 'LineWidth', 1.2, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_gtsam, 'MarkerFaceColor', 'w');

xlabel('Časový krok {\it k}', 'Interpreter', 'tex', 'FontSize', 14);
ylabel('Rychlost {\it v}_y [m/s]', 'Interpreter', 'tex', 'FontSize', 14);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;

% --- Hlavní nadpis a sdílená legenda ---
title(t_layout, '\bf Porovnání odhadů FG (MATLAB) a EKF', 'Interpreter', 'tex', 'FontSize', 16);

lgd = legend([h1, h2, h3], 'Referenční trajektorie', 'FG (MATLAB)', 'EKF', ...
    'Interpreter', 'tex', 'Orientation', 'horizontal');
lgd.FontSize = 12;
lgd.Layout.Tile = 'south'; % Umístí legendu krásně pod všechny 4 grafy

exportgraphics(fig_fg_comp, 'obrazky/TAN/FG_EKF_comp.pdf', 'ContentType', 'vector');


%% 3sigma intervaly pro FG a EKF
% fig_fg_comp_sigma = figure('Name', 'Porovnání MATLAB a GTSAM', 'Color', 'w', 'Position',[150, 150, 900, 700]);
% 
% % Vytvoření mřížky 2x2
% t_layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
% 
% % Zjištění počtu bodů pro rozložení značek (markerů)
% N_pts = size(estimated_states, 2);
% time_steps = 1:N_pts;
% 
% % --- VÝPOČET KONFIDENČNÍCH INTERVALŮ (3-sigma) PRO MATLAB FG ---
% sigma_bounds_fg = zeros(N_pts, 4);
% for i = 1:N_pts
%     if ~isempty(estimated_variances{i})
%         P_i_diag = diag(estimated_variances{i});
%         sigma_bounds_fg(i, :) = 3 * sqrt(max(P_i_diag, 1e-12))';
%     end
% end
% upper_bounds_fg = estimated_states + sigma_bounds_fg';
% lower_bounds_fg = estimated_states - sigma_bounds_fg';
% 
% % --- VÝPOČET KONFIDENČNÍCH INTERVALŮ (3-sigma) PRO EKF ---
% % Předpoklad: Kovariance z EKF jsou v proměnné "variances_EKF" (jako cell array).
% % Pokud máš kovariance EKF jako 3D matici (např. 4x4xN), změň řádek uvnitř 
% % smyčky na: P_i_diag_ekf = diag(variances_EKF(:,:,i));
% sigma_bounds_ekf = zeros(N_pts, 4);
% for i = 1:N_pts
%     if ~isempty(P_hist_EKF_cell{i}) 
%         P_i_diag_ekf = diag(P_hist_EKF_cell{i});
%         sigma_bounds_ekf(i, :) = 3 * sqrt(max(P_i_diag_ekf, 1e-12))';
%     end
% end
% upper_bounds_ekf = x_est_EKF + sigma_bounds_ekf';
% lower_bounds_ekf = x_est_EKF - sigma_bounds_ekf';
% 
% % X-ové souřadnice pro polygon (společné pro všechny grafy)
% fill_x = [time_steps, fliplr(time_steps)];
% 
% space = 50; % Rozestup mezi značkami na jedné čáře
% 
% % Fázově posunuté indexy (aby se značky nepřekrývaly)
% m_idx_true   = 1 : space : N_pts;
% m_idx_matlab = 11 : space : N_pts;
% m_idx_gtsam  = 22 : space : N_pts;
% 
% % --- 1. Poloha X ---
% nexttile;
% hold on; grid on;
% 
% % Vykreslení konfidenčního intervalu EKF (světle červená)
% fill_y1_ekf =[lower_bounds_ekf(1, :), fliplr(upper_bounds_ekf(1, :))];
% h_conf_ekf = fill(fill_x, fill_y1_ekf,[1.0 0.8 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
% 
% % Vykreslení konfidenčního intervalu FG (světle modrá)
% fill_y1_fg =[lower_bounds_fg(1, :), fliplr(upper_bounds_fg(1, :))];
% h_conf_fg = fill(fill_x, fill_y1_fg, [0.8 0.9 1.0], 'FaceAlpha', 0.5, 'EdgeColor', 'none'); 
% 
% % Uložíme handly pro legendu (čáry stavů)
% h1 = plot(x_true_all(1,:), 'k-', 'LineWidth', 1.5, ...
%     'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
% h2 = plot(estimated_states(1,:), 'b-', 'LineWidth', 1.5, ...
%     'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_matlab, 'MarkerFaceColor', 'w');
% h3 = plot(x_est_EKF(1,:), 'r-', 'LineWidth', 1.2, ...
%     'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_gtsam, 'MarkerFaceColor', 'w');
% 
% ylabel('Poloha {\it x}[m]', 'Interpreter', 'tex', 'FontSize', 14);
% ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
% xticklabels({}); % Skryje čísla na ose X pro horní graf
% 
% % --- 2. Poloha Y ---
% nexttile;
% hold on; grid on;
% 
% fill_y2_ekf =[lower_bounds_ekf(2, :), fliplr(upper_bounds_ekf(2, :))];
% fill(fill_x, fill_y2_ekf, [1.0 0.8 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
% 
% fill_y2_fg =[lower_bounds_fg(2, :), fliplr(upper_bounds_fg(2, :))];
% fill(fill_x, fill_y2_fg,[0.8 0.9 1.0], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
% 
% plot(x_true_all(2,:), 'k-', 'LineWidth', 1.5, ...
%     'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
% plot(estimated_states(2,:), 'b-', 'LineWidth', 1.5, ...
%     'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_matlab, 'MarkerFaceColor', 'w');
% plot(x_est_EKF(2,:), 'r-', 'LineWidth', 1.2, ...
%     'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_gtsam, 'MarkerFaceColor', 'w');
% 
% ylabel('Poloha {\it y} [m]', 'Interpreter', 'tex', 'FontSize', 14);
% ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
% xticklabels({}); % Skryje čísla na ose X pro horní graf
% 
% % --- 3. Rychlost X ---
% nexttile;
% hold on; grid on;
% 
% fill_y3_ekf =[lower_bounds_ekf(3, :), fliplr(upper_bounds_ekf(3, :))];
% fill(fill_x, fill_y3_ekf,[1.0 0.8 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
% 
% fill_y3_fg =[lower_bounds_fg(3, :), fliplr(upper_bounds_fg(3, :))];
% fill(fill_x, fill_y3_fg,[0.8 0.9 1.0], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
% 
% plot(x_true_all(3,:), 'k-', 'LineWidth', 1.5, ...
%     'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
% plot(estimated_states(3,:), 'b-', 'LineWidth', 1.5, ...
%     'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_matlab, 'MarkerFaceColor', 'w');
% plot(x_est_EKF(3,:), 'r-', 'LineWidth', 1.2, ...
%     'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_gtsam, 'MarkerFaceColor', 'w');
% 
% xlabel('Časový krok {\it k}', 'Interpreter', 'tex', 'FontSize', 14);
% ylabel('Rychlost {\it v}_x [m/s]', 'Interpreter', 'tex', 'FontSize', 14);
% ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
% 
% % --- 4. Rychlost Y ---
% nexttile;
% hold on; grid on;
% 
% fill_y4_ekf =[lower_bounds_ekf(4, :), fliplr(upper_bounds_ekf(4, :))];
% fill(fill_x, fill_y4_ekf,[1.0 0.8 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
% 
% fill_y4_fg =[lower_bounds_fg(4, :), fliplr(upper_bounds_fg(4, :))];
% fill(fill_x, fill_y4_fg,[0.8 0.9 1.0], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
% 
% plot(x_true_all(4,:), 'k-', 'LineWidth', 1.5, ...
%     'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', m_idx_true, 'MarkerFaceColor', 'w');
% plot(estimated_states(4,:), 'b-', 'LineWidth', 1.5, ...
%     'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', m_idx_matlab, 'MarkerFaceColor', 'w');
% plot(x_est_EKF(4,:), 'r-', 'LineWidth', 1.2, ...
%     'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', m_idx_gtsam, 'MarkerFaceColor', 'w');
% 
% xlabel('Časový krok {\it k}', 'Interpreter', 'tex', 'FontSize', 14);
% ylabel('Rychlost {\it v}_y [m/s]', 'Interpreter', 'tex', 'FontSize', 14);
% ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 12;
% 
% % --- Hlavní nadpis a sdílená legenda ---
% title(t_layout, '\bf Porovnání odhadů FG (MATLAB) a EKF', 'Interpreter', 'tex', 'FontSize', 16);
% 
% % Přidány handly pro oba polygonové intervaly do společné legendy
% lgd = legend([h1, h_conf_fg, h2, h_conf_ekf, h3], ...
%     'Referenční trajektorie', '3\sigma interval (FG)', 'FG (MATLAB)', ...
%     '3\sigma interval (EKF)', 'EKF', ...
%     'Interpreter', 'tex', 'Orientation', 'horizontal', 'NumColumns', 3);
% lgd.FontSize = 12;
% lgd.Layout.Tile = 'south'; % Umístí legendu krásně pod všechny 4 grafy
% 
% exportgraphics(fig_fg_comp_sigma, 'obrazky/TAN/FG_EKF_comp.pdf', 'ContentType', 'vector');
% 


% Vykreslení výsledků
% plot_results(N_total, x_true_all, estimated_states, estimated_variances, hB, map_m);


%% -------------------------------------------------------------------------
% Funkce pro vykreslení
% -------------------------------------------------------------------------
function plot_results(N, true_states, estimated_states, estimated_variances, hB, map_m)
time_steps = 1:N;

% 1. Grafy stavů s nejistotou
figure('Name', 'Porovnání stavů s vizualizací nejistoty pro FG (MATLAB)', 'Color', 'w');
state_names = ["Pozice X [m]", "Pozice Y [m]", "Rychlost Vx [m/s]", "Rychlost Vy [m/s]"];

sigma_bounds = zeros(N, 4);
for i = 1:N
    if ~isempty(estimated_variances{i})
        P_i_diag = diag(estimated_variances{i});
        sigma_bounds(i, :) = 3 * sqrt(max(P_i_diag, 1e-12))';
    end
end
upper = estimated_states + sigma_bounds';
lower = estimated_states - sigma_bounds';

for i = 1:4
    subplot(2, 2, i);
    fill_x = [time_steps, fliplr(time_steps)];
    fill_y = [lower(i, :), fliplr(upper(i, :))];

    fill(fill_x, fill_y, [0.8 0.9 0.8], 'FaceAlpha', 0.9, 'EdgeColor', 'none'); hold on;
    plot(time_steps, true_states(i, :), 'r-', 'LineWidth', 1.5);
    plot(time_steps, estimated_states(i, :), 'b--', 'LineWidth', 1.2);

    title(state_names(i)); grid on; axis tight;
    if i==1, legend('3\sigma Conf', 'True', 'FG est', 'Location', 'best'); end
end

% 2. 3D Mapa
% Nastavení velikosti okna (převzato ze šablony)
fig = figure('Name', 'Trajektorie na mapě', 'Color', 'w', 'Position',[200, 200, 1200, 600]);

% Vykreslení povrchu mapy
surf(map_m.x, map_m.y, map_m.z, 'EdgeColor', 'none', 'FaceAlpha', 1);
colormap turbo; 
shading interp; 
hold on; 

% Referenční trajektorie klasicky
plot3(true_states(1,:), true_states(2,:), hB(1:N), 'r-', 'LineWidth', 2);

% Odhadovaná s kroužky např. každých 50 kroků (číslo 50 upravte podle hustoty dat)
plot3(estimated_states(1,:), estimated_states(2,:), hB(1:N), 'b--', 'LineWidth', 1.5, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
    'MarkerIndices', 1:50:length(estimated_states));
% Oříznutí a poměr os (aby nebyla mapa placatá)
axis tight;
daspect([1 1 0.05]); 

% --- VIZUÁLNÍ STYLING PODLE ŠABLONY ---
grid on; 
ax = gca;
ax.GridAlpha = 0.3; % Průhlednější mřížka
ax.FontSize = 15;   % Velikost písma na osách (čísla)

% Popisky os
xlabel('X [m]', 'Interpreter', 'tex', 'FontSize', 17); 
ylabel('Y [m]', 'Interpreter', 'tex', 'FontSize', 17); 
zlabel('Výška [m]', 'Interpreter', 'tex', 'FontSize', 17);

% Legenda
lgd = legend('Terén', 'Referenční trajektorie', 'FG odhad (MATLAB)', ...
             'Interpreter', 'tex', 'Location', 'northeast');
lgd.FontSize = 15;

% Title (ve vašem původním kódu nebyl, ale přidávám zakomentovaný pro úplnost)
% title('Trajektorie letu nad terénem', 'Interpreter', 'tex', 'FontSize', 15);

% Export grafu do PDF
exportgraphics(fig, 'obrazky/TAN/3Dmapa.pdf', 'ContentType', 'image');
end