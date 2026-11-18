%% Sistemas Difusos Inteligentes.
%% Docente: Cristhian Torres Millarez.
%% Estudiante: Hernán Iván Montoya Motato - Nov 18, 2026
%% VALIDACIÓN DUAL EN MATLAB: RELACIONES, GRANULARIDAD Y FIS (MAMDANI VS SUGENO)
clear; clc; close all;

%% 1. ACTIVIDAD PRINCIPAL: COMPOSICIÓN DE RELACIONES DIFUSAS
R = [0.90 0.20 0.00; 0.30 0.80 0.20; 0.10 0.50 0.90];
S = [0.90 0.10 0.00; 0.20 0.70 0.30; 0.00 0.40 0.95];

[m, ~] = size(R); [~, p] = size(S);
T_maxmin = zeros(m,p); T_maxprod = zeros(m,p); T_sumprob = zeros(m,p);

for i = 1:m
    for j = 1:p
        v_prod = R(i,:) .* S(:,j)';
        T_maxmin(i,j)  = max(min(R(i,:), S(:,j)'));
        T_maxprod(i,j) = max(v_prod);
        T_sumprob(i,j) = 1 - prod(1 - v_prod); % s-norma: Suma Probabilística
    end
end
disp('--- Max-Min ---'); disp(round(T_maxmin, 3));
disp('--- Max-Producto ---'); disp(round(T_maxprod, 3));
disp('--- Suma Probabilística - Producto ---'); disp(round(T_sumprob, 3));

%% 2. INCISO B: SISTEMA MAMDANI EXHAUSTIVO Y CONVERSIÓN A SUGENO
fisM = mamfis('Name', 'ControlVentilacion_Mamdani');

fisM = addInput(fisM, [0 100], 'Name', 'Temperatura');
fisM = addMF(fisM, 'Temperatura', 'trapmf', [0 0 25 45], 'Name', 'Baja');
fisM = addMF(fisM, 'Temperatura', 'trimf',  [30 50 70],  'Name', 'Optima');
fisM = addMF(fisM, 'Temperatura', 'trapmf', [55 75 100 100], 'Name', 'Alta');

fisM = addInput(fisM, [0 100], 'Name', 'Carga_CPU');
fisM = addMF(fisM, 'Carga_CPU', 'trapmf', [0 0 25 45], 'Name', 'Baja');
fisM = addMF(fisM, 'Carga_CPU', 'trimf',  [30 50 70],  'Name', 'Media');
fisM = addMF(fisM, 'Carga_CPU', 'trapmf', [55 75 100 100], 'Name', 'Alta');

fisM = addOutput(fisM, [0 100], 'Name', 'Velocidad');
fisM = addMF(fisM, 'Velocidad', 'trimf', [0 0 45],    'Name', 'Baja');
fisM = addMF(fisM, 'Velocidad', 'trimf', [25 50 75],  'Name', 'Media');
fisM = addMF(fisM, 'Velocidad', 'trimf', [55 100 100],'Name', 'Alta');

reglas = [
    1 1 1 1 1; 1 2 1 1 1; 1 3 2 1 1;
    2 1 1 1 1; 2 2 2 1 1; 2 3 3 1 1;
    3 1 2 1 1; 3 2 3 1 1; 3 3 3 1 1
    ];
fisM = addRule(fisM, reglas);

% Conversión automática a Sugeno de orden 0 para contraste algorítmico
fisS = convertToSugeno(fisM);

entrada = [68 82];
fprintf('Salida Mamdani (Temp=68, CPU=82): %.2f%%\n', evalfis(fisM, entrada));
fprintf('Salida Sugeno  (Temp=68, CPU=82): %.2f%%\n', evalfis(fisS, entrada));

% Comparación visual de Superficies de Control 3D
figure('Name', 'Contraste Mamdani vs Sugeno', 'Color', 'w');
subplot(1,2,1); gensurf(fisM); title('Superficie Mamdani (Centroide)');
subplot(1,2,2); gensurf(fisS); title('Superficie Sugeno (Promedio Ponderado)');