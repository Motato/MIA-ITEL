%% Sistemas Difusos Inteligentes.
%% Docente: Cristhian Torres Millarez.
%% Estudiante: Hernán Iván Montoya Motato - Nov 22, 2026
function simulador_ev_tsukamoto() 
%% 1. UNIVERSOS Y FUNCIONES DE MEMBRESÍA
    x_soc  = linspace(0, 100, 201);
    y_temp = linspace(15, 65, 201);
    z_curr = linspace(0, 100, 401);

    s_bajo  = trapmf(x_soc, [0 0 20 40]);
    s_medio = trimf(x_soc,  [25 50 75]);
    s_alto  = trapmf(x_soc, [60 80 100 100]);

    t_norm = trapmf(y_temp, [15 15 25 35]);
    t_cal  = trimf(y_temp,  [30 40 50]);
    t_crit = trapmf(y_temp, [45 55 65 65]);

    c_baja  = trimf(z_curr, [0 0 40]);
    c_media = trimf(z_curr, [25 50 75]);
    c_alta  = trimf(z_curr, [60 100 100]);

    % FAM: [SoC(1=Bajo,2=Medio,3=Alto), Temp(1=Norm,2=Cal,3=Crit), Cons(1=Baja,2=Media,3=Alta)]
    FAM = [
        1 1 3;  1 2 2;  1 3 1;
        2 1 2;  2 2 2;  2 3 1;
        3 1 1;  3 2 1;  3 3 1
    ];

    % Precalcular curvas SISO 1D
    s_1d = linspace(0, 100, 101);
    c_t1d = zeros(size(s_1d)); c_m1d = zeros(size(s_1d)); c_s1d = zeros(size(s_1d));
    for i = 1:length(s_1d)
        [c_t1d(i), c_m1d(i), c_s1d(i)] = siso_eval(s_1d(i));
    end

    % Precalcular superficies MISO 3D (Caché 22x22)
    [S_g, T_g] = meshgrid(linspace(0, 100, 22), linspace(15, 65, 22));
    ZT = zeros(size(S_g)); ZM = zeros(size(S_g)); ZS = zeros(size(S_g));
    for i = 1:size(S_g, 1)
        for j = 1:size(S_g, 2)
            ZT(i,j) = miso_tsukamoto(S_g(i,j), T_g(i,j));
            ZM(i,j) = miso_mamdani(S_g(i,j), T_g(i,j));
            ZS(i,j) = miso_sugeno(S_g(i,j), T_g(i,j));
        end
    end

    %% 2. INTERFAZ GRÁFICA (UIFIGURE + PESTAÑAS)
    fig = uifigure('Name', 'Laboratorio EV: Tsukamoto, Mamdani y Sugeno (MATLAB)', ...
                   'Position', [50 40 1300 740]);
    tg  = uitabgroup(fig, 'Position', [10 10 1280 720]);

    tab1 = uitab(tg, 'Title', '1. SISO (1 Entrada): Tsukamoto vs. Mamdani y Sugeno');
    tab2 = uitab(tg, 'Title', '2. MISO (2 Entradas): Superficies 3D, Benchmark y Reglas');

    %% ==================== PESTAÑA 1: SISO ====================
    uilabel(tab1, 'Text', 'Estado de Carga SoC (%):', 'Position', [30 620 260 22], 'FontWeight', 'bold');
    sl_s1 = uislider(tab1, 'Position', [35 600 280 3], 'Limits', [0 100], 'Value', 30, ...
                     'ValueChangedFcn', @(~,~) actualizar_tab1());
    lbl_t1 = uilabel(tab1, 'Position', [30 510 300 35], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', [0.17 0.63 0.17]);
    lbl_m1 = uilabel(tab1, 'Position', [30 465 300 35], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', [0 0.45 0.74]);
    lbl_s1 = uilabel(tab1, 'Position', [30 420 300 35], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', [0.85 0.33 0.1]);
    ax1 = uiaxes(tab1, 'Position', [360 50 880 600]);

    %% ==================== PESTAÑA 2: MISO ====================
    uilabel(tab2, 'Text', 'Estado de Carga SoC (%):', 'Position', [30 650 200 22], 'FontWeight', 'bold');
    sl_s2 = uislider(tab2, 'Position', [35 635 540 3], 'Limits', [0 100], 'Value', 30, ...
                     'ValueChangedFcn', @(~,~) actualizar_tab2());

    uilabel(tab2, 'Text', 'Temperatura de Batería (°C):', 'Position', [650 650 220 22], 'FontWeight', 'bold');
    sl_t2 = uislider(tab2, 'Position', [655 635 540 3], 'Limits', [15 65], 'Value', 42, ...
                     'ValueChangedFcn', @(~,~) actualizar_tab2());

    lbl_t2 = uilabel(tab2, 'Position', [30 555 380 45],  'FontSize', 14, 'FontWeight', 'bold', 'FontColor', [0.17 0.63 0.17]);
    lbl_m2 = uilabel(tab2, 'Position', [440 555 380 45], 'FontSize', 14, 'FontWeight', 'bold', 'FontColor', [0 0.45 0.74]);
    lbl_s2 = uilabel(tab2, 'Position', [850 555 380 45], 'FontSize', 14, 'FontWeight', 'bold', 'FontColor', [0.85 0.33 0.1]);

    ax_t = uiaxes(tab2, 'Position', [20 210 400 340]);
    ax_m = uiaxes(tab2, 'Position', [435 210 400 340]);
    ax_s = uiaxes(tab2, 'Position', [850 210 400 340]);

    tbl_fam = uitable(tab2, 'Position', [20 15 1230 185], ...
        'ColumnName', {'Regla', 'Antecedentes (AND)', 'Consecuente', 'Fuerza de Disparo (α)'});

    %% ==================== MOTORES DE INFERENCIA ====================
    function [ot, om, os] = siso_eval(sv)
        a1 = interp1(x_soc, s_bajo, sv);
        a2 = interp1(x_soc, s_medio, sv);
        a3 = interp1(x_soc, s_alto, sv);
        den = max(a1 + a2 + a3, 1e-9);
        ot = (a1*(50+50*a1) + a2*(20+50*a2) + a3*(40*(1-a3))) / den;
        agg = max(min(a1, c_alta), max(min(a2, c_media), min(a3, c_baja)));
        om = defuzz(z_curr, agg, 'centroid');
        z1 = min(100, max(0, -0.4*sv + 95));
        z2 = min(100, max(0, -0.3*sv + 65));
        z3 = min(100, max(0, -0.2*sv + 30));
        os = (a1*z1 + a2*z2 + a3*z3) / den;
    end

    function alphas = eval_fam(sv, tv)
        ms = [interp1(x_soc, s_bajo, sv), interp1(x_soc, s_medio, sv), interp1(x_soc, s_alto, sv)];
        mt = [interp1(y_temp, t_norm, tv), interp1(y_temp, t_cal, tv), interp1(y_temp, t_crit, tv)];
        alphas = zeros(9, 2);
        for k = 1:9
            alphas(k, 1) = min(ms(FAM(k,1)), mt(FAM(k,2)));
            alphas(k, 2) = FAM(k, 3);
        end
    end

    function z_out = miso_tsukamoto(sv, tv)
        al = eval_fam(sv, tv); num = 0; den = 0;
        for k = 1:9
            a = al(k,1);
            if a > 0
                c = al(k,2);
                if c == 1,     zi = 40.0 * (1.0 - a);
                elseif c == 2, zi = 20.0 + 50.0 * a;
                else,          zi = 50.0 + 50.0 * a;
                end
                num = num + a*zi; den = den + a;
            end
        end
        z_out = num / max(den, 1e-9);
    end

    function z_out = miso_mamdani(sv, tv)
        al = eval_fam(sv, tv);
        agg = zeros(size(z_curr)); mfs = {c_baja, c_media, c_alta};
        for k = 1:9
            if al(k,1) > 0, agg = max(agg, min(al(k,1), mfs{al(k,2)})); end
        end
        z_out = 0;
        if sum(agg) > 0, z_out = defuzz(z_curr, agg, 'centroid'); end
    end

    function z_out = miso_sugeno(sv, tv)
        al = eval_fam(sv, tv); num = 0; den = 0;
        for k = 1:9
            a = al(k,1);
            if a > 0
                c = al(k,2);
                if c == 1,     zi = min(100, max(0, -0.2*sv - 0.3*tv + 35));
                elseif c == 2, zi = min(100, max(0, -0.3*sv - 0.5*tv + 80));
                else,          zi = min(100, max(0, -0.4*sv - 0.6*tv + 115));
                end
                num = num + a*zi; den = den + a;
            end
        end
        z_out = num / max(den, 1e-9);
    end

    %% ==================== CALLBACKS ====================
    function actualizar_tab1()
        sv = sl_s1.Value;
        [ot, om, os] = siso_eval(sv);
        lbl_t1.Text = sprintf('Tsukamoto 1D: %.2f A', ot);
        lbl_m1.Text = sprintf('Mamdani 1D:   %.2f A', om);
        lbl_s1.Text = sprintf('Sugeno 1D:    %.2f A', os);

        cla(ax1); hold(ax1, 'on');
        plot(ax1, s_1d, c_t1d, 'Color', [0.17 0.63 0.17], 'LineWidth', 2.5, 'DisplayName', 'Tsukamoto (Monótono)');
        plot(ax1, s_1d, c_m1d, '--', 'Color', [0 0.45 0.74], 'LineWidth', 2.2, 'DisplayName', 'Mamdani (Centroide)');
        plot(ax1, s_1d, c_s1d, ':', 'Color', [0.85 0.33 0.1], 'LineWidth', 2.5, 'DisplayName', 'Sugeno (1.er Orden)');
        plot(ax1, [sv sv sv], [ot om os], 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Punto Actual');
        hold(ax1, 'off'); grid(ax1, 'on'); legend(ax1, 'Location', 'northeast');
        title(ax1, 'Curvas de Transferencia SISO (1 Entrada → 1 Salida)');
        xlabel(ax1, 'Estado de Carga SoC (%)'); ylabel(ax1, 'Corriente de Carga (A)');
    end

    function actualizar_tab2()
        sv = sl_s2.Value; tv = sl_t2.Value;
        tic; ot = miso_tsukamoto(sv, tv); tt = toc * 1000;
        tic; om = miso_mamdani(sv, tv);   tm = toc * 1000;
        tic; os = miso_sugeno(sv, tv);    ts = toc * 1000;

        lbl_t2.Text = sprintf('Tsukamoto (Monótono): %.2f A\nTiempo: %.3f ms', ot, tt);
        lbl_m2.Text = sprintf('Mamdani (Centroide): %.2f A\nTiempo: %.3f ms', om, tm);
        lbl_s2.Text = sprintf('Sugeno (1.er Orden): %.2f A\nTiempo: %.3f ms', os, ts);

        graficar_3d(ax_t, ZT, ot, sv, tv, 'Superficie Tsukamoto');
        graficar_3d(ax_m, ZM, om, sv, tv, 'Superficie Mamdani');
        graficar_3d(ax_s, ZS, os, sv, tv, 'Superficie Sugeno (TSK)');

        al = eval_fam(sv, tv);
        nS = {'Bajo', 'Medio', 'Alto'}; nT = {'Normal', 'Cálida', 'Crítica'}; nC = {'Baja', 'Media', 'Alta'};
        datos = cell(9, 4);
        for k = 1:9
            datos{k,1} = sprintf('R%d', k);
            datos{k,2} = [nS{FAM(k,1)} ' ∧ ' nT{FAM(k,2)}];
            datos{k,3} = nC{FAM(k,3)};
            datos{k,4} = round(al(k,1), 3);
        end
        tbl_fam.Data = datos;
    end

    function graficar_3d(ax, Z_mat, z_pt, sv, tv, titulo)
        cla(ax); hold(ax, 'on');
        surf(ax, S_g, T_g, Z_mat, 'EdgeColor', 'interp', 'FaceAlpha', 0.9);
        plot3(ax, sv, tv, z_pt, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        hold(ax, 'off'); view(ax, 45, 30); grid(ax, 'on');
        title(ax, titulo); xlabel(ax, 'SoC (%)'); ylabel(ax, 'Temp (°C)'); zlabel(ax, 'Corr (A)');
    end

    actualizar_tab1();
    actualizar_tab2();
end