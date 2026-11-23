%% Sistemas Difusos Inteligentes.
%% Docente: Cristhian Torres Millarez.
%% Estudiante: Hernán Iván Montoya Motato - Nov 22, 2026
function simulador_acc_sugeno()
  
%% 1. UNIVERSOS Y FUNCIONES DE MEMBRESÍA
    x_dist = linspace(0, 100, 201);
    y_vrel = linspace(-30, 30, 201);
    z_ctrl = linspace(-100, 100, 401);

    d_corta = trapmf(x_dist, [0 0 20 40]);
    d_media = trimf(x_dist,  [25 50 75]);
    d_larga = trapmf(x_dist, [60 80 100 100]);

    v_alej = trapmf(y_vrel, [-30 -30 -15 0]);
    v_ig   = trimf(y_vrel,  [-10 0 10]);
    v_acer = trapmf(y_vrel, [0 15 30 30]);

    c_fren = trimf(z_ctrl, [-100 -100 -10]);
    c_mant = trimf(z_ctrl, [-30 0 30]);
    c_acel = trimf(z_ctrl, [10 100 100]);

    % FAM: [Dist(1=Corta,2=Media,3=Larga), Vrel(1=Alej,2=Ig,3=Acer), Cons(1=Fren,2=Mant,3=Acel)]
    FAM = [
        1 1 2;  1 2 1;  1 3 1;
        2 1 3;  2 2 2;  2 3 1;
        3 1 3;  3 2 3;  3 3 2
    ];

    % Precalcular curvas SISO 1D
    d_1d = linspace(0, 100, 101);
    curva_s1d = zeros(size(d_1d)); curva_m1d = zeros(size(d_1d));
    for i = 1:length(d_1d)
        [curva_s1d(i), curva_m1d(i)] = siso_eval(d_1d(i));
    end

    % Precalcular superficies MISO 3D (Caché 22x22)
    [D_g, V_g] = meshgrid(linspace(0, 100, 22), linspace(-30, 30, 22));
    ZS = zeros(size(D_g)); ZM = zeros(size(D_g)); ZT = zeros(size(D_g));
    for i = 1:size(D_g, 1)
        for j = 1:size(D_g, 2)
            ZS(i,j) = miso_sugeno(D_g(i,j), V_g(i,j));
            ZM(i,j) = miso_mamdani(D_g(i,j), V_g(i,j));
            ZT(i,j) = miso_tsukamoto(D_g(i,j), V_g(i,j));
        end
    end

    %% 2. INTERFAZ GRÁFICA (UIFIGURE + PESTAÑAS)
    fig = uifigure('Name', 'Laboratorio ACC: Sugeno, Mamdani y Tsukamoto (MATLAB)', ...
                   'Position', [50 40 1300 740]);
    tg  = uitabgroup(fig, 'Position', [10 10 1280 720]);

    tab1 = uitab(tg, 'Title', '1. SISO (1 Entrada): Sugeno vs. Mamdani');
    tab2 = uitab(tg, 'Title', '2. MISO (2 Entradas): Sugeno vs. Mamdani vs. Tsukamoto');

    %% ==================== PESTAÑA 1: SISO ====================
    uilabel(tab1, 'Text', 'Distancia al vehículo frontal (m):', 'Position', [30 620 260 22], 'FontWeight', 'bold');
    sl_d1 = uislider(tab1, 'Position', [35 600 280 3], 'Limits', [0 100], 'Value', 32, ...
                     'ValueChangedFcn', @(~,~) actualizar_tab1());
    lbl_s1 = uilabel(tab1, 'Position', [30 500 300 40], 'FontSize', 16, 'FontWeight', 'bold', 'FontColor', [0.85 0.33 0.1]);
    lbl_m1 = uilabel(tab1, 'Position', [30 450 300 40], 'FontSize', 16, 'FontWeight', 'bold', 'FontColor', [0 0.45 0.74]);
    ax1 = uiaxes(tab1, 'Position', [360 50 880 600]);

    %% ==================== PESTAÑA 2: MISO ====================
    uilabel(tab2, 'Text', 'Distancia (m):', 'Position', [30 650 150 22], 'FontWeight', 'bold');
    sl_d2 = uislider(tab2, 'Position', [35 635 540 3], 'Limits', [0 100], 'Value', 32, ...
                     'ValueChangedFcn', @(~,~) actualizar_tab2());

    uilabel(tab2, 'Text', 'Velocidad Relativa (km/h):', 'Position', [650 650 200 22], 'FontWeight', 'bold');
    sl_v2 = uislider(tab2, 'Position', [655 635 540 3], 'Limits', [-30 30], 'Value', 12, ...
                     'ValueChangedFcn', @(~,~) actualizar_tab2());

    lbl_s2 = uilabel(tab2, 'Position', [30 555 380 45],  'FontSize', 14, 'FontWeight', 'bold', 'FontColor', [0.85 0.33 0.1]);
    lbl_m2 = uilabel(tab2, 'Position', [440 555 380 45], 'FontSize', 14, 'FontWeight', 'bold', 'FontColor', [0 0.45 0.74]);
    lbl_t2 = uilabel(tab2, 'Position', [850 555 380 45], 'FontSize', 14, 'FontWeight', 'bold', 'FontColor', [0.47 0.67 0.19]);

    ax_s = uiaxes(tab2, 'Position', [20 210 400 340]);
    ax_m = uiaxes(tab2, 'Position', [435 210 400 340]);
    ax_t = uiaxes(tab2, 'Position', [850 210 400 340]);

    tbl_fam = uitable(tab2, 'Position', [20 15 1230 185], ...
        'ColumnName', {'Regla', 'Antecedentes (AND)', 'Consecuente', 'Fuerza de Disparo (α)'});

    %% ==================== MOTORES DE INFERENCIA ====================
    function [os, om] = siso_eval(dv)
        ac = interp1(x_dist, d_corta, dv);
        am = interp1(x_dist, d_media, dv);
        al = interp1(x_dist, d_larga, dv);
        z1 = min(100, max(-100, 0.8*dv - 90));
        z2 = min(100, max(-100, 0.4*dv - 20));
        z3 = min(100, max(-100, 0.6*dv + 15));
        os = (ac*z1 + am*z2 + al*z3) / max(ac + am + al, 1e-9);
        agg = max(min(ac, c_fren), max(min(am, c_mant), min(al, c_acel)));
        om = defuzz(z_ctrl, agg, 'centroid');
    end

    function alphas = eval_fam(dv, vv)
        md = [interp1(x_dist, d_corta, dv), interp1(x_dist, d_media, dv), interp1(x_dist, d_larga, dv)];
        mv = [interp1(y_vrel, v_alej, vv),  interp1(y_vrel, v_ig, vv),    interp1(y_vrel, v_acer, vv)];
        alphas = zeros(9, 2);
        for k = 1:9
            alphas(k, 1) = min(md(FAM(k,1)), mv(FAM(k,2)));
            alphas(k, 2) = FAM(k, 3);
        end
    end

    function z_out = miso_mamdani(dv, vv)
        al = eval_fam(dv, vv);
        agg = zeros(size(z_ctrl)); mfs = {c_fren, c_mant, c_acel};
        for k = 1:9
            if al(k,1) > 0, agg = max(agg, min(al(k,1), mfs{al(k,2)})); end
        end
        z_out = 0;
        if sum(agg) > 0, z_out = defuzz(z_ctrl, agg, 'centroid'); end
    end

    function z_out = miso_sugeno(dv, vv)
        al = eval_fam(dv, vv); num = 0; den = 0;
        for k = 1:9
            a = al(k,1);
            if a > 0
                c = al(k,2);
                if c == 1,     zi = min(100, max(-100, 0.5*dv - 1.2*vv - 60));
                elseif c == 2, zi = min(100, max(-100, 0.2*dv - 0.5*vv));
                else,          zi = min(100, max(-100, 0.6*dv - 0.8*vv + 20));
                end
                num = num + a*zi; den = den + a;
            end
        end
        z_out = num / max(den, 1e-9);
    end

    function z_out = miso_tsukamoto(dv, vv)
        al = eval_fam(dv, vv); num = 0; den = 0;
        for k = 1:9
            a = al(k,1);
            if a > 0
                c = al(k,2);
                if c == 1,     zi = -100.0 * a;
                elseif c == 2, zi = -25.0 + 50.0 * a;
                else,          zi = 100.0 * a;
                end
                num = num + a*zi; den = den + a;
            end
        end
        z_out = num / max(den, 1e-9);
    end

    %% ==================== CALLBACKS DE ACTUALIZACIÓN ====================
    function actualizar_tab1()
        dv = sl_d1.Value;
        [os, om] = siso_eval(dv);
        lbl_s1.Text = sprintf('Sugeno 1D:  %.2f %%', os);
        lbl_m1.Text = sprintf('Mamdani 1D: %.2f %%', om);

        cla(ax1); hold(ax1, 'on');
        plot(ax1, d_1d, curva_s1d, 'Color', [0.85 0.33 0.1], 'LineWidth', 2.5, 'DisplayName', 'Sugeno (1.er Orden)');
        plot(ax1, d_1d, curva_m1d, '--', 'Color', [0 0.45 0.74], 'LineWidth', 2.5, 'DisplayName', 'Mamdani (Centroide)');
        plot(ax1, [dv dv], [os om], 'ro', 'MarkerSize', 9, 'MarkerFaceColor', 'r', 'DisplayName', 'Punto Actual');
        hold(ax1, 'off'); grid(ax1, 'on'); legend(ax1, 'Location', 'northwest');
        title(ax1, 'Curvas de Transferencia SISO (Negativo = Frenar | Positivo = Acelerar)');
        xlabel(ax1, 'Distancia (m)'); ylabel(ax1, 'Acción de Control (%)');
    end

    function actualizar_tab2()
        dv = sl_d2.Value; vv = sl_v2.Value;
        tic; os = miso_sugeno(dv, vv);    ts = toc * 1000;
        tic; om = miso_mamdani(dv, vv);   tm = toc * 1000;
        tic; ot = miso_tsukamoto(dv, vv); tt = toc * 1000;

        lbl_s2.Text = sprintf('Sugeno (1.er Orden): %.2f %%\nTiempo: %.3f ms', os, ts);
        lbl_m2.Text = sprintf('Mamdani (Centroide): %.2f %%\nTiempo: %.3f ms', om, tm);
        lbl_t2.Text = sprintf('Tsukamoto (Monótono): %.2f %%\nTiempo: %.3f ms', ot, tt);

        graficar_3d(ax_s, ZS, os, dv, vv, 'Superficie Sugeno (TSK)');
        graficar_3d(ax_m, ZM, om, dv, vv, 'Superficie Mamdani');
        graficar_3d(ax_t, ZT, ot, dv, vv, 'Superficie Tsukamoto');

        al = eval_fam(dv, vv);
        nD = {'Corta', 'Media', 'Larga'}; nV = {'Alejándose', 'Igual', 'Acercándose'}; nC = {'Frenar', 'Mantener', 'Acelerar'};
        datos = cell(9, 4);
        for k = 1:9
            datos{k,1} = sprintf('R%d', k);
            datos{k,2} = [nD{FAM(k,1)} ' ∧ ' nV{FAM(k,2)}];
            datos{k,3} = nC{FAM(k,3)};
            datos{k,4} = round(al(k,1), 3);
        end
        tbl_fam.Data = datos;
    end

    function graficar_3d(ax, Z_mat, z_pt, dv, vv, titulo)
        cla(ax); hold(ax, 'on');
        surf(ax, D_g, V_g, Z_mat, 'EdgeColor', 'interp', 'FaceAlpha', 0.9);
        plot3(ax, dv, vv, z_pt, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
        hold(ax, 'off'); view(ax, 45, 30); grid(ax, 'on');
        title(ax, titulo); xlabel(ax, 'Dist (m)'); ylabel(ax, 'VelRel'); zlabel(ax, 'Ctrl (%)');
    end
    actualizar_tab1();
    actualizar_tab2();
end