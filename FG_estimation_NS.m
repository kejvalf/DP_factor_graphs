%% ========================================================================
% Spouštěcí skript pro FG estimaci - NS
% =========================================================================
clc; clear;
close all;
rng(8);

% ========================================================================
%  DEFINICE MODELU
% =========================================================================
model.nx = 2;
model.nz = 1;

model.H = @(x) [x(1)];
% Jacobián měření H(x)
model.dHdX = @(x) [1, 0];

% Dynamika  F - nelineární
model.F = @(x) [x(1) * x(2);
                  x(2)    ];
% Jakobidán dynamiky  F
model.dFdX =@(x) [x(2) ,x(1);
                  0    ,1   ];

% --- Matice šumu pro FG ESTIMÁTOR (Řešič) ---
q_1 = 1e-3;   
q_2 = 1e-10;  % malý šum -> x2 je téměř konstanta
Q = diag([q_1, q_2]);

r1 = 1e-2;    % šum měření stavu x1
R = diag(r1);

%% PŘÍPRAVA DAT (Generování)
N = 100;
x0 =[1; 1.0005]; % Skutečný počátek (např. mírný růst o 0.5% za krok)
P0 = diag([1.0, 10]); % Prior (X1 známe celkem dobře, konstantu moc ne)

% Skutečný šum pro vygenerování dat (Ground truth)
% x2 je v realitě skutečná konstanta -> její šum je 0
Q_true = diag([1e-5, 0]); 

meas = zeros(model.nz, N);
x_true = zeros(model.nx, N+1); 
x_true(:,1) = x0;

% Simulace systému
for i = 1:N
    % Krok skutečné dynamiky
    x_true(:,i+1) = model.F(x_true(:,i)) + sqrt(Q_true) * randn(2,1);
    
    % Generování měření s přesným senzorem
    meas(:,i) = model.H(x_true(:,i)) + sqrt(R) * randn(1,1);
end
x_true = x_true(:, 1:N); % Oříznutí na správnou délku


%% opt FG
estimated_states = zeros(model.nx, N);
estimated_variances_FG = cell(1, N);
solver = FactorGraphSolver(model, meas, R, Q, x0, P0);
tic
solver.opt();
toc
estimated_states = solver.states';
% Výpočet a uložení kovariance
P_full = solver.compute_covariance();
for t = 1:N
    idx_s = (t-1)*model.nx + 1;
    idx_e = t*model.nx;
    P_t = full(P_full(idx_s:idx_e, idx_s:idx_e));
    estimated_variances_FG{t} = P_t;
end


%% opt EKF
myFilter = TrajectoryFilters(model, Q, R, x0, P0);
tic
[x_est_EKF, P_hist_EKF] = myFilter.runEKF(meas);
toc
for i = 1:size(P_hist_EKF,3)
    P_hist_EKF_cell{i} = P_hist_EKF(:,:,i);
end

%% opt ERTSS
tic
[x_est_ERTSS, P_hist_ERTSS] = myFilter.runRTSS(x_est_EKF, P_hist_EKF);
toc
for i = 1:size(P_hist_ERTSS,3)
    P_hist_ERTSS_cell{i} = P_hist_ERTSS(:,:,i);
end

%% VYHODNOCENÍ
RMSE_FG = sqrt(mean((estimated_states - x_true(:,1:N)).^2, 2));
RMSE_EKF = sqrt(mean((x_est_EKF - x_true).^2, 2));
RMSE_ERTSS = sqrt(mean((x_est_ERTSS - x_true).^2, 2));

format long
fprintf('Výsledné RMSE EKF:\n');
disp(RMSE_EKF)
fprintf('Výsledné RMSE ERTSS:\n');
disp(RMSE_ERTSS)
fprintf('Výsledné RMSE FG-MATLAB:\n');
disp(RMSE_FG)

fprintf('EKF:\n');
ANEES_EKF = ANEES(x_est_EKF,x_true,P_hist_EKF_cell,model);
fprintf('ERTSS:\n');
ANEES_ERTSS = ANEES(x_est_ERTSS,x_true,P_hist_ERTSS_cell,model);
fprintf('FG-MATLAB:\n');
ANEES_FG = ANEES(estimated_states,x_true,estimated_variances_FG,model);



%% ========================================================================
% Vykreslení výsledků s 3-sigma intervaly pomocí tiledlayout 
% =========================================================================

time_steps = 1:N;

% --- 1. Extrakce 3-sigma mezí z kovariančních matic ---
sigma_FG    = zeros(2, N);
sigma_EKF   = zeros(2, N);
sigma_ERTSS = zeros(2, N);

for t = 1:N
    % FG
    sigma_FG(1, t) = 3 * sqrt(estimated_variances_FG{t}(1,1));
    sigma_FG(2, t) = 3 * sqrt(estimated_variances_FG{t}(2,2));
    
    % EKF
    sigma_EKF(1, t) = 3 * sqrt(P_hist_EKF(1,1,t));
    sigma_EKF(2, t) = 3 * sqrt(P_hist_EKF(2,2,t));

    % ERTSS
    sigma_ERTSS(1, t) = 3 * sqrt(P_hist_ERTSS(1,1,t));
    sigma_ERTSS(2, t) = 3 * sqrt(P_hist_ERTSS(2,2,t));
end

% --- 2. Vykreslení pomocí tiledlayout ---
fig = figure('Name', 'Porovnání s 3-sigma', 'Color', 'w', 'Position',[200, 200, 900, 700]);

% Uspořádání 2x1
t_layout = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

state_names_tex = {'Poloha {\it x}_1 [m]', 'Neznámá konstanta {\it x}_2 [-]'};

% Fázově posunuté indexy značek pro černobílý tisk (aby se neslily)
space = 8; 
idx_true  = 1 : space : N;
idx_ekf   = 3 : space : N;
idx_ertss = 5 : space : N;
idx_fg    = 7 : space : N;

for i = 1:2
    nexttile; 
    hold on; grid on;

    % --- A) Oblasti nejistoty (poloprůhledné pásy) ---
    fill_x = [time_steps, fliplr(time_steps)];
    
    % 1) EKF (světle modrá)
    fill_y_ekf =[(x_est_EKF(i,:) - sigma_EKF(i,:)), fliplr(x_est_EKF(i,:) + sigma_EKF(i,:))];
    h_fill_ekf = fill(fill_x, fill_y_ekf,[0.8 0.8 1.0], 'FaceAlpha', 0.8, 'EdgeColor', 'none');

    % 2) ERTSS (světle zelená)
    fill_y_ertss =[(x_est_ERTSS(i,:) - sigma_ERTSS(i,:)), fliplr(x_est_ERTSS(i,:) + sigma_ERTSS(i,:))];
    h_fill_ertss = fill(fill_x, fill_y_ertss, [0.8 1.0 0.8], 'FaceAlpha', 1.0, 'EdgeColor', 'none');

    % 3) FG (světle červená)
    fill_y_fg =[(estimated_states(i,:) - sigma_FG(i,:)), fliplr(estimated_states(i,:) + sigma_FG(i,:))];
    h_fill_fg = fill(fill_x, fill_y_fg, [1.0 0.8 0.8], 'FaceAlpha', 0.8, 'EdgeColor', 'none');

    % --- B) Trajektorie s markery ---
    % EKF - modrá čárkovaná, čtverečky
    h_ekf = plot(time_steps, x_est_EKF(i,:), 'b--', 'LineWidth', 1.5, ...
        'Marker', 's', 'MarkerSize', 5, 'MarkerIndices', idx_ekf, 'MarkerFaceColor', 'w');
    
    % ERTSS - zelená tečkovaná, křížky
    h_ertss = plot(time_steps, x_est_ERTSS(i,:), 'g:', 'LineWidth', 2.0, ...
        'Marker', 'x', 'MarkerSize', 6, 'MarkerIndices', idx_ertss);
    
    % FG - červená čerchovaná, trojúhelníky
    h_fg = plot(time_steps, estimated_states(i,:), 'r-.', 'LineWidth', 1.5, ...
        'Marker', '^', 'MarkerSize', 5, 'MarkerIndices', idx_fg, 'MarkerFaceColor', 'w');

    % Referenční - černá plná, kolečka (Vykresleno nakonec, aby byla nahoře)
    h_true = plot(time_steps, x_true(i,:), 'k-', 'LineWidth', 1.5, ...
        'Marker', 'o', 'MarkerSize', 5, 'MarkerIndices', idx_true, 'MarkerFaceColor', 'w');

    % --- C) Formátování os ---
    xlim([0 N]);
    if i == 1 
        ylim([0.5 1.5])
    end
    if i == 2 
        ylim([0.85 1.15])
    end
    
    ylabel(state_names_tex{i}, 'Interpreter', 'tex', 'FontSize', 15);
    ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 13;

    % Osu X dáme jen pod spodní graf
    if i == 2
        xlabel('Časový krok {\it k}', 'Interpreter', 'tex', 'FontSize', 15);
    else
        xticklabels({}); 
    end
end

% --- 3. Hlavní nadpis a legenda ---
title(t_layout, '\bf Porovnání odhadů a nejistot (3\sigma) pro nelineární systém', 'Interpreter', 'tex', 'FontSize', 16);

% Společná legenda vložená bezpečně dolů pomocí Tile = 'south'
lgd = legend([h_true, h_ekf, h_fill_ekf, h_ertss, h_fill_ertss, h_fg, h_fill_fg], ...
    'Referenční', 'EKF', 'EKF 3\sigma', 'ERTSS', 'ERTSS 3\sigma', 'FG', 'FG 3\sigma', ...
    'Interpreter', 'tex', 'NumColumns', 4); % 4 sloupce, aby se to hezky rozložilo
lgd.FontSize = 12;
lgd.Layout.Tile = 'south'; 

exportgraphics(fig, 'obrazky/NS/porovnani.pdf', 'ContentType', 'vector');