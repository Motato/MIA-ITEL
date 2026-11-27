function agronash_fis_proyecto()
%% Sistemas Difusos Inteligentes.
%% Docente: Cristhian Torres Millarez.
%% Estudiante: Hernán Iván Montoya Motato - Nov 26, 2026
%% 1. CAPA 1: CONSTRUCCIÓN DEL SISTEMA MAMDANI (3 ENTRADAS, 27 REGLAS)
    fis = mamfis('Name', 'AgroNash_FIS');

    fis = addInput(fis, [0 100], 'Name', 'Estres');
    fis = addMF(fis, 'Estres', 'trapmf', [0 0 20 40],     'Name', 'Bajo');
    fis = addMF(fis, 'Estres', 'trimf',  [25 50 75],      'Name', 'Moderado');
    fis = addMF(fis, 'Estres', 'trapmf', [60 80 100 100], 'Name', 'Severo');

    fis = addInput(fis, [0 100], 'Name', 'Disponibilidad');
    fis = addMF(fis, 'Disponibilidad', 'trapmf', [0 0 20 40],     'Name', 'Critica');
    fis = addMF(fis, 'Disponibilidad', 'trimf',  [25 50 75],      'Name', 'Media');
    fis = addMF(fis, 'Disponibilidad', 'trapmf', [60 80 100 100], 'Name', 'Alta');

    fis = addInput(fis, [0 100], 'Name', 'Eficiencia');
    fis = addMF(fis, 'Eficiencia', 'trapmf', [0 0 30 50],     'Name', 'Baja');
    fis = addMF(fis, 'Eficiencia', 'trimf',  [40 60 80],      'Name', 'Media');
    fis = addMF(fis, 'Eficiencia', 'trapmf', [70 85 100 100], 'Name', 'Alta');

    fis = addOutput(fis, [0 100], 'Name', 'Cuota');
    fis = addMF(fis, 'Cuota', 'trimf', [0 0 35],     'Name', 'Minima');
    fis = addMF(fis, 'Cuota', 'trimf', [20 50 80],   'Name', 'Moderada');
    fis = addMF(fis, 'Cuota', 'trimf', [65 100 100], 'Name', 'Optima');

    % Generación sistemática de las 27 reglas (1=Minima, 2=Moderada, 3=Optima)
    reglas = zeros(27, 6);
    idx = 1;
    for e = 1:3
        for d = 1:3
            for f = 1:3
                if e == 1
                    sal = 1;
                elseif e == 2
                    if d == 1, sal = 1;
                    elseif d == 3 && f == 3, sal = 3;
                    else, sal = 2;
                    end
                else % e == 3 (Severo)
                    if d == 1 && f == 1, sal = 1;
                    elseif d == 1 || (d == 2 && f == 1), sal = 2;
                    else, sal = 3;
                    end
                end
                reglas(idx, :) = [e, d, f, sal, 1, 1]; % [E, D, F, Salida, Peso, 1=AND]
                idx = idx + 1;
            end
        end
    end
    fis = addRule(fis, reglas);

    %% 2. INTERFAZ GRÁFICA (UIFIGURE + 3 PESTAÑAS)
    fig = uifigure('Name', 'AgroNash-FIS | Proyecto Final Lógica Difusa (MATLAB)', ...
                   'Position', [60 50 1240 700]);
    tg  = uitabgroup(fig, 'Position', [10 10 1220 680]);

    tab1 = uitab(tg, 'Title', '1. Simulador Mamdani (3 Entradas)');
    tab2 = uitab(tg, 'Title', '2. Innovación: Negociación Difusa de Nash');
    tab3 = uitab(tg, 'Title', '3. Validación Automática de Casos');

    %% ==================== PESTAÑA 1: MAMDANI 3D ====================
    uilabel(tab1, 'Text', 'Estrés Hídrico (%):', 'Position', [25 600 200 22], 'FontWeight', 'bold');
    sl_e = uislider(tab1, 'Position', [30 585 280 3], 'Limits', [0 100], 'Value', 85, ...
                    'ValueChangedFcn', @(~,~) actualizar_tab1());

    uilabel(tab1, 'Text', 'Disponibilidad Acuífero (%):', 'Position', [25 520 220 22], 'FontWeight', 'bold');
    sl_d = uislider(tab1, 'Position', [30 505 280 3], 'Limits', [0 100], 'Value', 55, ...
                    'ValueChangedFcn', @(~,~) actualizar_tab1());

    uilabel(tab1, 'Text', 'Eficiencia de Riego (%):', 'Position', [25 440 220 22], 'FontWeight', 'bold');
    sl_f = uislider(tab1, 'Position', [30 425 280 3], 'Limits', [0 100], 'Value', 90, ...
                    'ValueChangedFcn', @(~,~) actualizar_tab1());

    lbl_cuota = uilabel(tab1, 'Position', [25 320 300 60], 'FontSize', 18, ...
                        'FontWeight', 'bold', 'FontColor', [0 0.45 0.74]);
    ax1 = uiaxes(tab1, 'Position', [360 40 820 580]);

    %% ==================== PESTAÑA 2: FUZZY NASH ====================
    uilabel(tab2, 'Text', 'Agua Total Disponible B (m³):', 'Position', [25 600 220 22], 'FontWeight', 'bold');
    sl_B = uislider(tab2, 'Position', [30 585 280 3], 'Limits', [500 2000], 'Value', 1000, ...
                    'ValueChangedFcn', @(~,~) actualizar_tab2());

    uilabel(tab2, 'Text', 'Zona A - Mínimo d1 (m³):', 'Position', [25 515 180 22]);
    ed_d1 = uieditfield(tab2, 'numeric', 'Position', [210 515 90 22], 'Value', 200, 'ValueChangedFcn', @(~,~) actualizar_tab2());
    uilabel(tab2, 'Text', 'Zona A - Ideal (m³):', 'Position', [25 480 180 22]);
    ed_i1 = uieditfield(tab2, 'numeric', 'Position', [210 480 90 22], 'Value', 700, 'ValueChangedFcn', @(~,~) actualizar_tab2());

    uilabel(tab2, 'Text', 'Zona B - Mínimo d2 (m³):', 'Position', [25 430 180 22]);
    ed_d2 = uieditfield(tab2, 'numeric', 'Position', [210 430 90 22], 'Value', 150, 'ValueChangedFcn', @(~,~) actualizar_tab2());
    uilabel(tab2, 'Text', 'Zona B - Ideal (m³):', 'Position', [25 395 180 22]);
    ed_i2 = uieditfield(tab2, 'numeric', 'Position', [210 395 90 22], 'Value', 600, 'ValueChangedFcn', @(~,~) actualizar_tab2());

    lbl_nash = uilabel(tab2, 'Position', [25 260 310 110], 'FontSize', 13, 'FontWeight', 'bold', 'VerticalAlignment', 'top');
    ax2 = uiaxes(tab2, 'Position', [360 40 820 580]);

    %% ==================== PESTAÑA 3: VALIDACIÓN ====================
    casos = {
        '1. Sequía extrema + Riego ineficiente', 85, 15, 30, 'Mínima (<30%)';
        '2. Estrés severo + Riego por goteo',    85, 55, 90, 'Óptima (>75%)';
        '3. Cultivo sano + Acuífero lleno',      15, 85, 70, 'Mínima (<30%)';
        '4. Condición promedio de temporada',    50, 50, 60, 'Moderada (~50%)'
    };
    datos_val = cell(4, 6);
    for k = 1:4
        out_v = evalfis(fis, [casos{k,2}, casos{k,3}, casos{k,4}]);
        datos_val(k, :) = {casos{k,1}, casos{k,2}, casos{k,3}, casos{k,4}, casos{k,5}, round(out_v, 2)};
    end
    uitable(tab3, 'Position', [30 220 1150 380], 'Data', datos_val, ...
            'ColumnName', {'Escenario', 'Estrés (%)', 'Acuífero (%)', 'Eficiencia (%)', 'Esperado', 'Obtenido (%)'});

    %% ==================== CALLBACKS ====================
    function actualizar_tab1()
        ev = sl_e.Value; dv = sl_d.Value; fv = sl_f.Value;
        cuota = evalfis(fis, [ev dv fv]);
        lbl_cuota.Text = sprintf('Cuota Recomendada:\n%.2f %%', cuota);

        [E_m, D_m] = meshgrid(linspace(0, 100, 18), linspace(0, 100, 18));
        F_m = repmat(fv, size(E_m));
        Z_m = reshape(evalfis(fis, [E_m(:), D_m(:), F_m(:)]), size(E_m));

        cla(ax1); hold(ax1, 'on');
        surf(ax1, E_m, D_m, Z_m, 'EdgeColor', 'interp', 'FaceAlpha', 0.9);
        plot3(ax1, ev, dv, cuota, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        hold(ax1, 'off'); view(ax1, 45, 30); grid(ax1, 'on');
        title(ax1, sprintf('Superficie de Decisión (Eficiencia fija en %.0f%%)', fv));
        xlabel(ax1, 'Estrés (%)'); ylabel(ax1, 'Acuífero (%)'); zlabel(ax1, 'Cuota (%)');
    end

    function actualizar_tab2()
        B = sl_B.Value;
        d1 = ed_d1.Value; i1 = ed_i1.Value;
        d2 = ed_d2.Value; i2 = ed_i2.Value;

        x1 = linspace(0, B, 1001);
        x2 = B - x1;
        mu1 = min(1, max(0, (x1 - d1) / max(i1 - d1, 1e-6)));
        mu2 = min(1, max(0, (x2 - d2) / max(i2 - d2, 1e-6)));

        nash_prod = mu1 .* mu2;
        bz_min    = min(mu1, mu2);

        [~, idx_n] = max(nash_prod);
        [~, idx_b] = max(bz_min);

        lbl_nash.Text = sprintf( ...
            'Fuzzy Nash (Producto):\n  A = %.0f m³ | B = %.0f m³\n  (μ1=%.2f, μ2=%.2f)\n\nBellman-Zadeh (Mínimo):\n  A = %.0f m³ | B = %.0f m³', ...
            x1(idx_n), x2(idx_n), mu1(idx_n), mu2(idx_n), x1(idx_b), x2(idx_b));

        cla(ax2); hold(ax2, 'on');
        plot(ax2, x1, mu1, ':', 'LineWidth', 1.5, 'DisplayName', 'Satisfacción Zona A (μ1)');
        plot(ax2, x1, mu2, ':', 'LineWidth', 1.5, 'DisplayName', 'Satisfacción Zona B (μ2)');
        plot(ax2, x1, nash_prod, 'Color', [0 0.6 0.2], 'LineWidth', 2.8, 'DisplayName', 'Fuzzy Nash (μ1 · μ2)');
        plot(ax2, x1, bz_min, 'Color', [0.5 0 0.7], 'LineWidth', 2.5, 'DisplayName', 'Bellman-Zadeh min(μ1, μ2)');
        hold(ax2, 'off'); grid(ax2, 'on'); legend(ax2, 'Location', 'north');
        title(ax2, 'Comparativa de Negociación Difusa');
        xlabel(ax2, 'Agua Asignada a Zona A (m³)'); ylabel(ax2, 'Grado de Satisfacción / Decisión');
    end

    actualizar_tab1();
    actualizar_tab2();
end