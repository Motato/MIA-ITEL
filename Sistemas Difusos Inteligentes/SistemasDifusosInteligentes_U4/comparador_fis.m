%% Sistemas Difusos Inteligentes.
%% Docente: Cristhian Torres Millarez.
%% Estudiante: Hernán Iván Montoya Motato - Nov 19, 2026
%% Tema: Modelo de Inferencia difusa - Mamdani
function comparador_fis()
    %% 1. UNIVERSOS DE DISCURSO Y FUNCIONES DE MEMBRESÍA COMPARTIDAS
    x_temp = linspace(15, 35, 201);
    y_hum  = linspace(20, 90, 201);
    z_pot  = linspace(0, 100, 201);

    % Membresías de Temperatura [15, 35] °C
    t_fria = trapmf(x_temp, [15 15 19 23]);
    t_conf = trimf(x_temp,  [20 24 28]);
    t_cal  = trapmf(x_temp, [25 29 35 35]);

    % Membresías de Humedad [20, 90] %
    h_seca = trapmf(y_hum, [20 20 35 50]);
    h_norm = trimf(y_hum,  [35 55 75]);
    h_hum  = trapmf(y_hum, [60 75 90 90]);

    % Membresías de Salida Mamdani [0, 100] %
    p_baja  = trimf(z_pot, [0 0 45]);
    p_media = trimf(z_pot, [25 50 75]);
    p_alta  = trimf(z_pot, [55 100 100]);

    % Matriz FAM (9 reglas): [Temp(1=Fria,2=Conf,3=Cal), Hum(1=Seca,2=Norm,3=Hum), Cons(1=Baja,2=Media,3=Alta)]
    FAM = [
        1 1 1;  1 2 1;  1 3 2;
        2 1 1;  2 2 2;  2 3 3;
        3 1 2;  3 2 3;  3 3 3
    ];

    % Datos para Pestaña 1 (Mamdani SISO 1D)
    err_u = linspace(-10, 10, 201);
    e_neg = trapmf(err_u, [-10 -10 -4 0]);
    e_cer = trimf(err_u,  [-3 0 3]);
    e_pos = trapmf(err_u, [0 4 10 10]);

    % Precalcular curva de transferencia 1D
    err_sample = err_u(1:4:end);
    curva_1d = zeros(size(err_sample));
    for i = 1:length(err_sample)
        an = interp1(err_u, e_neg, err_sample(i));
        ac = interp1(err_u, e_cer, err_sample(i));
        ap = interp1(err_u, e_pos, err_sample(i));
        ag = max(min(an, p_baja), max(min(ac, p_media), min(ap, p_alta)));
        curva_1d(i) = defuzz(z_pot, ag, 'centroid');
    end

    % Precalcular superficies 3D (malla 22x22 en caché)
    [T_g, H_g] = meshgrid(linspace(15, 35, 22), linspace(20, 90, 22));
    ZM = zeros(size(T_g)); ZS = zeros(size(T_g)); ZT = zeros(size(T_g));
    for i = 1:size(T_g, 1)
        for j = 1:size(T_g, 2)
            ZM(i,j) = inferir_mamdani(T_g(i,j), H_g(i,j));
            ZS(i,j) = inferir_sugeno(T_g(i,j), H_g(i,j));
            ZT(i,j) = inferir_tsukamoto(T_g(i,j), H_g(i,j));
        end
    end

    %% 2. INTERFAZ GRÁFICA (UIFIGURE + PESTAÑAS)
    fig = uifigure('Name', 'Laboratorio Comparativo FIS: Mamdani, Sugeno y Tsukamoto (HVAC)', ...
                   'Position', [60 60 1280 700]);
    tg  = uitabgroup(fig, 'Position', [10 10 1260 680]);

    tab1 = uitab(tg, 'Title', '1. Mamdani SISO (1 Entrada → 1 Salida)');
    tab2 = uitab(tg, 'Title', '2. Comparativa MISO: Mamdani vs. Sugeno vs. Tsukamoto');

    %% ==================== PESTAÑA 1: MAMDANI SISO ====================
    uilabel(tab1, 'Text', 'Error de Temperatura (T_actual - T_deseada) °C:', ...
            'Position', [30 580 320 22], 'FontWeight', 'bold');
    sl_err = uislider(tab1, 'Position', [35 560 300 3], 'Limits', [-10 10], 'Value', 3.5, ...
                      'ValueChangedFcn', @(~,~) actualizar_tab1());

    lbl_pot1d = uilabel(tab1, 'Position', [30 460 320 50], 'FontSize', 18, ...
                        'FontWeight', 'bold', 'FontColor', [0 0.5 0.5]);
    lbl_act1d = uilabel(tab1, 'Position', [30 370 320 75], 'FontSize', 13, ...
                        'VerticalAlignment', 'top');
    ax1d = uiaxes(tab1, 'Position', [380 50 840 560]);

    %% ==================== PESTAÑA 2: COMPARATIVA MISO ====================
    uilabel(tab2, 'Text', 'Temperatura Interior (°C):', 'Position', [40 610 200 22], 'FontWeight', 'bold');
    sl_t = uislider(tab2, 'Position', [45 595 520 3], 'Limits', [15 35], 'Value', 27.5, ...
                    'ValueChangedFcn', @(~,~) actualizar_tab2());

    uilabel(tab2, 'Text', 'Humedad Relativa (%):', 'Position', [660 610 200 22], 'FontWeight', 'bold');
    sl_h = uislider(tab2, 'Position', [665 595 520 3], 'Limits', [20 90], 'Value', 68.0, ...
                    'ValueChangedFcn', @(~,~) actualizar_tab2());

    % Etiquetas de métricas y tiempos de ejecución (Benchmark)
    lbl_m = uilabel(tab2, 'Position', [40 495 360 55],  'FontSize', 15, 'FontWeight', 'bold', 'FontColor', [0 0.45 0.74]);
    lbl_s = uilabel(tab2, 'Position', [450 495 360 55], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', [0.85 0.33 0.1]);
    lbl_t = uilabel(tab2, 'Position', [860 495 360 55], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', [0.47 0.67 0.19]);

    % Tres ejes 3D lado a lado
    ax_m = uiaxes(tab2, 'Position', [20 30 400 450]);
    ax_s = uiaxes(tab2, 'Position', [430 30 400 450]);
    ax_t = uiaxes(tab2, 'Position', [840 30 400 450]);

    %% ==================== MOTORES DE INFERENCIA ====================
    function alphas = evaluar_antecedentes(t_val, h_val)
        mt = [interp1(x_temp, t_fria, t_val), interp1(x_temp, t_conf, t_val), interp1(x_temp, t_cal, t_val)];
        mh = [interp1(y_hum, h_seca, h_val),  interp1(y_hum, h_norm, h_val),  interp1(y_hum, h_hum, h_val)];
        alphas = zeros(9, 2);
        for k = 1:9
            alphas(k, 1) = min(mt(FAM(k,1)), mh(FAM(k,2)));
            alphas(k, 2) = FAM(k, 3);
        end
    end

    function z_out = inferir_mamdani(t_val, h_val)
        alphas = evaluar_antecedentes(t_val, h_val);
        agg = zeros(size(z_pot));
        mfs = {p_baja, p_media, p_alta};
        for k = 1:9
            a = alphas(k, 1);
            if a > 0
                agg = max(agg, min(a, mfs{alphas(k, 2)}));
            end
        end
        z_out = 0;
        if sum(agg) > 0
            z_out = defuzz(z_pot, agg, 'centroid');
        end
    end

    function z_out = inferir_sugeno(t_val, h_val)
        alphas = evaluar_antecedentes(t_val, h_val);
        num = 0; den = 0;
        for k = 1:9
            a = alphas(k, 1);
            if a > 0
                c = alphas(k, 2);
                if c == 1,     zi = min(100, max(0, 0.5*t_val + 0.1*h_val));
                elseif c == 2, zi = min(100, max(0, 1.2*t_val + 0.3*h_val + 5.0));
                else,          zi = min(100, max(0, 2.0*t_val + 0.4*h_val + 15.0));
                end
                num = num + a * zi;
                den = den + a;
            end
        end
        z_out = num / max(den, 1e-9);
    end

    function z_out = inferir_tsukamoto(t_val, h_val)
        alphas = evaluar_antecedentes(t_val, h_val);
        num = 0; den = 0;
        for k = 1:9
            a = alphas(k, 1);
            if a > 0
                c = alphas(k, 2);
                if c == 1,     zi = 50.0 * (1.0 - a); % Monótona decreciente [0, 50]
                elseif c == 2, zi = 25.0 + 50.0 * a;  % Monótona creciente [25, 75]
                else,          zi = 50.0 + 50.0 * a;  % Monótona creciente [50, 100]
                end
                num = num + a * zi;
                den = den + a;
            end
        end
        z_out = num / max(den, 1e-9);
    end

    %% ==================== CALLBACKS DE ACTUALIZACIÓN ====================
    function actualizar_tab1()
        e = sl_err.Value;
        an = interp1(err_u, e_neg, e);
        ac = interp1(err_u, e_cer, e);
        ap = interp1(err_u, e_pos, e);
        ag = max(min(an, p_baja), max(min(ac, p_media), min(ap, p_alta)));
        pot = defuzz(z_pot, ag, 'centroid');

        lbl_pot1d.Text = sprintf('Potencia Compresor: %.2f %%', pot);
        lbl_act1d.Text = sprintf('Activaciones:\n• Negativo: %.2f\n• Cero: %.2f\n• Positivo: %.2f', an, ac, ap);

        cla(ax1d); hold(ax1d, 'on');
        plot(ax1d, err_sample, curva_1d, 'Color', [0 0.5 0.5], 'LineWidth', 2.5, 'DisplayName', 'Curva Mamdani');
        plot(ax1d, e, pot, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Punto Actual');
        hold(ax1d, 'off'); grid(ax1d, 'on'); legend(ax1d, 'Location', 'northwest');
        title(ax1d, 'Curva de Transferencia No Lineal (1 Entrada → 1 Salida)');
        xlabel(ax1d, 'Error de Temperatura (°C)'); ylabel(ax1d, 'Potencia (%)');
    end

    function actualizar_tab2()
        tv = sl_t.Value; hv = sl_h.Value;

        % Benchmark de tiempos en milisegundos
        tic; out_m = inferir_mamdani(tv, hv);   tm = toc * 1000;
        tic; out_s = inferir_sugeno(tv, hv);    ts = toc * 1000;
        tic; out_t = inferir_tsukamoto(tv, hv); tt = toc * 1000;

        lbl_m.Text = sprintf('Mamdani (Centroide): %.2f %%\nTiempo: %.3f ms', out_m, tm);
        lbl_s.Text = sprintf('Sugeno (1.er Orden): %.2f %%\nTiempo: %.3f ms', out_s, ts);
        lbl_t.Text = sprintf('Tsukamoto (Monótono): %.2f %%\nTiempo: %.3f ms', out_t, tt);

        graficar_3d(ax_m, ZM, out_m, tv, hv, 'Superficie Mamdani');
        graficar_3d(ax_s, ZS, out_s, tv, hv, 'Superficie Sugeno (TSK)');
        graficar_3d(ax_t, ZT, out_t, tv, hv, 'Superficie Tsukamoto');
    end

    function graficar_3d(ax, Z_mat, z_pt, tv, hv, titulo)
        cla(ax); hold(ax, 'on');
        surf(ax, T_g, H_g, Z_mat, 'EdgeColor', 'interp', 'FaceAlpha', 0.9);
        plot3(ax, tv, hv, z_pt, 'ro', 'MarkerSize', 9, 'MarkerFaceColor', 'r');
        hold(ax, 'off'); view(ax, 45, 30); grid(ax, 'on');
        title(ax, titulo); xlabel(ax, 'Temp (°C)'); ylabel(ax, 'Hum (%)'); zlabel(ax, 'Pot (%)');
    end

    % Inicializar ambas pestañas
    actualizar_tab1();
    actualizar_tab2();
end