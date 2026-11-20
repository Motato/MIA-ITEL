%% Sistemas Difusos Inteligentes.
%% Docente: Cristhian Torres Millarez.
%% Estudiante: Hernán Iván Montoya Motato - Nov 19, 2026
%% Tema: Modelo de Inferencia Difusa.
function simulador_inferencia()
    %% 1. UNIVERSOS DE DISCURSO Y FUNCIONES DE MEMBRESÍA BASE
    x_hum  = linspace(0, 100, 501); % Humedad del suelo (%)
    y_temp = linspace(0, 45, 451);  % Temperatura ambiente (°C)
    z_rieg = linspace(0, 60, 601);  % Duración de riego (min)

    hum_seca   = trapmf(x_hum, [0 0 25 45]);
    hum_optima = trimf(x_hum,  [30 50 70]);
    hum_humeda = trapmf(x_hum, [55 75 100 100]);

    temp_baja     = trapmf(y_temp, [0 0 12 20]);
    temp_templada = trimf(y_temp,  [14 24 34]);
    temp_alta     = trapmf(y_temp, [26 35 45 45]);

    rieg_corto = trimf(z_rieg, [0 0 25]);
    rieg_medio = trimf(z_rieg, [15 30 45]);
    rieg_largo = trimf(z_rieg, [35 60 60]);

    %% 2. CONSTRUCCIÓN DEL SISTEMA MAMDANI EXHAUSTIVO (9 REGLAS) Y CACHÉ 3D
    fis = mamfis('Name', 'RiegoInvernadero');
    fis = addInput(fis, [0 100], 'Name', 'Humedad');
    fis = addMF(fis, 'Humedad', 'trapmf', [0 0 25 45],     'Name', 'Seca');
    fis = addMF(fis, 'Humedad', 'trimf',  [30 50 70],      'Name', 'Optima');
    fis = addMF(fis, 'Humedad', 'trapmf', [55 75 100 100], 'Name', 'Humeda');

    fis = addInput(fis, [0 45], 'Name', 'Temperatura');
    fis = addMF(fis, 'Temperatura', 'trapmf', [0 0 12 20],   'Name', 'Baja');
    fis = addMF(fis, 'Temperatura', 'trimf',  [14 24 34],    'Name', 'Templada');
    fis = addMF(fis, 'Temperatura', 'trapmf', [26 35 45 45], 'Name', 'Alta');

    fis = addOutput(fis, [0 60], 'Name', 'Riego');
    fis = addMF(fis, 'Riego', 'trimf', [0 0 25],   'Name', 'Corto');
    fis = addMF(fis, 'Riego', 'trimf', [15 30 45], 'Name', 'Medio');
    fis = addMF(fis, 'Riego', 'trimf', [35 60 60], 'Name', 'Largo');

    reglas = [
        1 1 2 1 1;  1 2 3 1 1;  1 3 3 1 1;
        2 1 1 1 1;  2 2 2 1 1;  2 3 3 1 1;
        3 1 1 1 1;  3 2 1 1 1;  3 3 2 1 1
    ];
    fis = addRule(fis, reglas);

    % Precalcular malla 3D una sola vez (equivalente a @st.cache_data)
    [H_grid, T_grid] = meshgrid(linspace(0, 100, 26), linspace(0, 45, 26));
    Z_grid = reshape(evalfis(fis, [H_grid(:), T_grid(:)]), size(H_grid));

    %% 3. VENTANA PRINCIPAL Y PESTAÑAS (EQUIVALENTE A STREAMLIT)
    fig = uifigure('Name', 'Laboratorio Interactivo de Lógica Difusa (MATLAB)', ...
                   'Position', [80 60 1220 700]);
    tg  = uitabgroup(fig, 'Position', [10 10 1200 680]);

    tab1 = uitab(tg, 'Title', '1. Inferencia Composicional (Casos a, b, c)');
    tab2 = uitab(tg, 'Title', '2. Evaluación a): Granularidad (3, 5, 7)');
    tab3 = uitab(tg, 'Title', '3. Evaluación b): Sistema Exhaustivo y 3D');

    %% ==================== PESTAÑA 1: INFERENCIA CRI ====================
    uilabel(tab1, 'Text', 'Esquema de Reglas:', 'Position', [20 600 180 22], 'FontWeight', 'bold');
    dd_caso = uidropdown(tab1, 'Position', [20 570 300 28], ...
        'Items', {'Caso a) 1 Regla, 1 Antecedente', ...
                  'Caso b) 1 Regla, Múltiples Antecedentes', ...
                  'Caso c) Múltiples Reglas, Múltiples Antecedentes'}, ...
        'ValueChangedFcn', @(~,~) actualizar_tab1());

    uilabel(tab1, 'Text', 'Operador de Implicación:', 'Position', [20 525 180 22], 'FontWeight', 'bold');
    dd_imp = uidropdown(tab1, 'Position', [20 495 300 28], ...
        'Items', {'Mamdani (Recorte min)', 'Larsen (Escalado producto)'}, ...
        'ValueChangedFcn', @(~,~) actualizar_tab1());

    uilabel(tab1, 'Text', 'Humedad del Suelo (%):', 'Position', [20 445 200 22], 'FontWeight', 'bold');
    sl_h1 = uislider(tab1, 'Position', [25 430 290 3], 'Limits', [0 100], 'Value', 35, ...
        'ValueChangedFcn', @(~,~) actualizar_tab1());

    uilabel(tab1, 'Text', 'Temperatura Ambiente (°C):', 'Position', [20 365 200 22], 'FontWeight', 'bold');
    sl_t1 = uislider(tab1, 'Position', [25 350 290 3], 'Limits', [0 45], 'Value', 31, ...
        'ValueChangedFcn', @(~,~) actualizar_tab1());

    txt_info1 = uitextarea(tab1, 'Position', [20 150 300 160], 'Editable', 'off');
    lbl_defuzz = uilabel(tab1, 'Position', [20 40 300 95], 'FontWeight', 'bold', ...
        'FontSize', 13, 'VerticalAlignment', 'top');

    ax1 = uiaxes(tab1, 'Position', [350 40 820 580]);

    %% ==================== PESTAÑA 2: GRANULARIDAD ====================
    uilabel(tab2, 'Text', 'Número de Valores Difusos:', 'Position', [20 600 220 22], 'FontWeight', 'bold');
    dd_gran = uidropdown(tab2, 'Position', [20 570 300 28], ...
        'Items', {'Caso 1: 3 Valores (Trapezoidal/Triangular)', ...
                  'Caso 2: 5 Valores (Gaussianas)', ...
                  'Caso 3: 7 Valores (Campana Generalizada)'}, ...
        'ValueChangedFcn', @(~,~) actualizar_tab2());

    uilabel(tab2, 'Text', 'Humedad de Prueba (%):', 'Position', [20 510 200 22], 'FontWeight', 'bold');
    sl_h2 = uislider(tab2, 'Position', [25 495 290 3], 'Limits', [0 100], 'Value', 42, ...
        'ValueChangedFcn', @(~,~) actualizar_tab2());

    uilabel(tab2, 'Text', 'Umbral de Corte α (Alpha-cut):', 'Position', [20 430 220 22], 'FontWeight', 'bold');
    sl_a2 = uislider(tab2, 'Position', [25 415 290 3], 'Limits', [0 1], 'Value', 0.5, ...
        'ValueChangedFcn', @(~,~) actualizar_tab2());

    tbl_gran = uitable(tab2, 'Position', [20 40 300 330], ...
        'ColumnName', {'Valor', 'μ(x)', '¿≥ α?'});
    ax2 = uiaxes(tab2, 'Position', [350 40 820 580]);

    %% ==================== PESTAÑA 3: SISTEMA EXHAUSTIVO 3D ====================
    uilabel(tab3, 'Text', 'Humedad del Suelo (%):', 'Position', [20 600 200 22], 'FontWeight', 'bold');
    sl_h3 = uislider(tab3, 'Position', [25 585 290 3], 'Limits', [0 100], 'Value', 35, ...
        'ValueChangedFcn', @(~,~) actualizar_tab3());

    uilabel(tab3, 'Text', 'Temperatura Ambiente (°C):', 'Position', [20 520 200 22], 'FontWeight', 'bold');
    sl_t3 = uislider(tab3, 'Position', [25 505 290 3], 'Limits', [0 45], 'Value', 31, ...
        'ValueChangedFcn', @(~,~) actualizar_tab3());

    lbl_out3 = uilabel(tab3, 'Position', [20 420 300 45], 'FontSize', 16, ...
        'FontWeight', 'bold', 'FontColor', [0 0.45 0.74]);

    tbl_fam = uitable(tab3, 'Position', [20 40 320 360], ...
        'ColumnName', {'Regla', 'Antecedentes', 'Salida', 'Disparo α'});
    ax3 = uiaxes(tab3, 'Position', [370 40 800 580]);

    %% ==================== CALLBACKS DE ACTUALIZACIÓN ====================
    function actualizar_tab1()
        h = sl_h1.Value; t = sl_t1.Value;
        mh_s = interp1(x_hum, hum_seca, h);
        mh_o = interp1(x_hum, hum_optima, h);
        mh_h = interp1(x_hum, hum_humeda, h);
        mt_b = interp1(y_temp, temp_baja, t);
        mt_t = interp1(y_temp, temp_templada, t);
        mt_a = interp1(y_temp, temp_alta, t);

        es_mamdani = contains(dd_imp.Value, 'Mamdani');
        imp = @(a, mf) (es_mamdani * min(a, mf)) + (~es_mamdani * (a .* mf));

        if contains(dd_caso.Value, 'Caso a)')
            a1 = mh_s;
            agg = imp(a1, rieg_largo);
            txt_info1.Value = {sprintf('Humedad = %.1f%% | Temp = %.1f°C', h, t), ...
                               'Regla: SI Seca ENTONCES Largo', ...
                               sprintf('Activación α1 = %.3f', a1)};
        elseif contains(dd_caso.Value, 'Caso b)')
            a1 = min(mh_s, mt_a);
            agg = imp(a1, rieg_largo);
            txt_info1.Value = {sprintf('Humedad = %.1f%% | Temp = %.1f°C', h, t), ...
                               'Regla: SI Seca Y Alta ENTONCES Largo', ...
                               sprintf('Activación α1 = min(%.2f, %.2f) = %.3f', mh_s, mt_a, a1)};
        else
            a1 = min(mh_s, mt_a);
            a2 = min(mh_o, mt_t);
            a3 = min(mh_h, mt_b);
            agg = max(imp(a1, rieg_largo), max(imp(a2, rieg_medio), imp(a3, rieg_corto)));
            txt_info1.Value = {sprintf('Humedad = %.1f%% | Temp = %.1f°C', h, t), ...
                               sprintf('R1 (Seca ∧ Alta → Largo): α1 = %.3f', a1), ...
                               sprintf('R2 (Óptima ∧ Temp → Medio): α2 = %.3f', a2), ...
                               sprintf('R3 (Húmeda ∧ Baja → Corto): α3 = %.3f', a3)};
        end

        if sum(agg) == 0
            zc = 0; zb = 0; zm = 0;
        else
            zc = defuzz(z_rieg, agg, 'centroid');
            zb = defuzz(z_rieg, agg, 'bisector');
            zm = defuzz(z_rieg, agg, 'mom');
        end
        lbl_defuzz.Text = sprintf('Centroide: %.2f min\nBisector:  %.2f min\nMOM:       %.2f min', zc, zb, zm);

        cla(ax1); hold(ax1, 'on');
        plot(ax1, z_rieg, rieg_corto, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
        plot(ax1, z_rieg, rieg_medio, ':', 'Color', [0.85 0.33 0.1], 'LineWidth', 1.2);
        plot(ax1, z_rieg, rieg_largo, ':', 'Color', [0 0.45 0.74], 'LineWidth', 1.2);
        area(ax1, z_rieg, agg, 'FaceColor', [0 0.5 0.5], 'FaceAlpha', 0.35, 'EdgeColor', [0 0.5 0.5], 'LineWidth', 2.5);
        if zc > 0, xline(ax1, zc, '--r', sprintf('Centroide = %.2f', zc), 'LineWidth', 1.8); end
        hold(ax1, 'off'); ylim(ax1, [-0.05 1.1]); grid(ax1, 'on');
        title(ax1, ['Salida Inferida — ' dd_caso.Value]);
        xlabel(ax1, 'Duración del Riego (minutos)'); ylabel(ax1, 'Grado de Membresía \mu(z)');
    end

    function actualizar_tab2()
        h = sl_h2.Value; alpha = sl_a2.Value;
        cla(ax2); hold(ax2, 'on');
        if contains(dd_gran.Value, '3 Valores')
            nombres = {'Seco', 'Óptimo', 'Húmedo'};
            mfs = {hum_seca, hum_optima, hum_humeda};
        elseif contains(dd_gran.Value, '5 Valores')
            nombres = {'Muy Seco', 'Seco', 'Óptimo', 'Húmedo', 'Saturado'};
            centros = [10 30 50 70 90];
            mfs = arrayfun(@(c) gaussmf(x_hum, [10 c]), centros, 'UniformOutput', false);
        else
            nombres = {'Árido', 'Muy Seco', 'Seco', 'Óptimo', 'Húmedo', 'Muy Húmedo', 'Saturado'};
            centros = linspace(5, 95, 7);
            mfs = arrayfun(@(c) gbellmf(x_hum, [8 2.5 c]), centros, 'UniformOutput', false);
        end

        datos = cell(length(nombres), 3);
        for k = 1:length(nombres)
            plot(ax2, x_hum, mfs{k}, 'LineWidth', 2, 'DisplayName', nombres{k});
            val = interp1(x_hum, mfs{k}, h);
            datos{k, 1} = nombres{k};
            datos{k, 2} = round(val, 4);
            datos{k, 3} = ternary(val >= alpha, 'Sí', 'No');
        end
        xline(ax2, h, '--k', sprintf('x = %.1f%%', h), 'LineWidth', 1.5, 'HandleVisibility', 'off');
        yline(ax2, alpha, ':r', sprintf('α = %.2f', alpha), 'LineWidth', 1.5, 'HandleVisibility', 'off');
        hold(ax2, 'off'); ylim(ax2, [-0.05 1.1]); grid(ax2, 'on'); legend(ax2, 'Location', 'best');
        title(ax2, ['Humedad del Suelo — ' dd_gran.Value]);
        xlabel(ax2, 'Humedad (%)'); ylabel(ax2, 'Grado de Membresía \mu(x)');
        tbl_gran.Data = datos;
    end

    function actualizar_tab3()
        h = sl_h3.Value; t = sl_t3.Value;
        r_out = evalfis(fis, [h t]);
        lbl_out3.Text = sprintf('Riego Óptimo: %.2f min', r_out);

        mh = [interp1(x_hum, hum_seca, h), interp1(x_hum, hum_optima, h), interp1(x_hum, hum_humeda, h)];
        mt = [interp1(y_temp, temp_baja, t), interp1(y_temp, temp_templada, t), interp1(y_temp, temp_alta, t)];
        nH = {'Seca', 'Óptima', 'Húmeda'}; nT = {'Baja', 'Templada', 'Alta'}; nR = {'Corto', 'Medio', 'Largo'};

        datos_fam = cell(9, 4);
        for k = 1:9
            iH = reglas(k,1); iT = reglas(k,2); iR = reglas(k,3);
            datos_fam{k,1} = sprintf('R%d', k);
            datos_fam{k,2} = [nH{iH} ' ∧ ' nT{iT}];
            datos_fam{k,3} = nR{iR};
            datos_fam{k,4} = round(min(mh(iH), mt(iT)), 3);
        end
        tbl_fam.Data = datos_fam;

        cla(ax3); hold(ax3, 'on');
        surf(ax3, H_grid, T_grid, Z_grid, 'FaceAlpha', 0.88, 'EdgeColor', 'interp');
        plot3(ax3, h, t, r_out, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        hold(ax3, 'off'); view(ax3, 45, 30); grid(ax3, 'on');
        xlabel(ax3, 'Humedad (%)'); ylabel(ax3, 'Temperatura (°C)'); zlabel(ax3, 'Riego (min)');
        title(ax3, 'Superficie de Control 3D (Humedad × Temperatura → Riego)');
    end

    function out = ternary(cond, a, b)
        if cond, out = a; else, out = b; end
    end

    % Inicializar las tres pestañas al abrir la ventana
    actualizar_tab1(); actualizar_tab2(); actualizar_tab3();
end