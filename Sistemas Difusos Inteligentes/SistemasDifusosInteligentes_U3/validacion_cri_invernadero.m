%% Sistemas Difusos Inteligentes.
%% Docente: Cristhian Torres Millarez.
%% Estudiante: Hernán Iván Montoya Motato - Nov 19, 2026
%% VALIDACIÓN DUAL EN MATLAB: CONTROLADOR DE RIEGO (MAMDANI VS SUGENO)
clear; clc; close all;

% 1. Crear sistema Mamdani (2 Entradas, 1 Salida)
fisM = mamfis('Name', 'RiegoInvernadero');

fisM = addInput(fisM, [0 100], 'Name', 'Humedad');
fisM = addMF(fisM, 'Humedad', 'trapmf', [0 0 25 45],     'Name', 'Seca');
fisM = addMF(fisM, 'Humedad', 'trimf',  [30 50 70],      'Name', 'Optima');
fisM = addMF(fisM, 'Humedad', 'trapmf', [55 75 100 100], 'Name', 'Humeda');

fisM = addInput(fisM, [0 45], 'Name', 'Temperatura');
fisM = addMF(fisM, 'Temperatura', 'trapmf', [0 0 12 20],  'Name', 'Baja');
fisM = addMF(fisM, 'Temperatura', 'trimf',  [14 24 34],   'Name', 'Templada');
fisM = addMF(fisM, 'Temperatura', 'trapmf', [26 35 45 45],'Name', 'Alta');

fisM = addOutput(fisM, [0 60], 'Name', 'Riego');
fisM = addMF(fisM, 'Riego', 'trimf', [0 0 25],   'Name', 'Corto');
fisM = addMF(fisM, 'Riego', 'trimf', [15 30 45], 'Name', 'Medio');
fisM = addMF(fisM, 'Riego', 'trimf', [35 60 60], 'Name', 'Largo');

% 2. Base de reglas exhaustiva (3x3 = 9 combinaciones: [Hum, Temp, Riego, Peso, 1=AND])
reglas = [
    1 1 2 1 1;  1 2 3 1 1;  1 3 3 1 1;
    2 1 1 1 1;  2 2 2 1 1;  2 3 3 1 1;
    3 1 1 1 1;  3 2 1 1 1;  3 3 2 1 1
    ];
fisM = addRule(fisM, reglas);

% 3. Conversión a Sugeno para contraste metodológico
fisS = convertToSugeno(fisM);

% 4. Evaluación en el mismo punto de prueba (Humedad = 35%, Temp = 31°C)
entrada = [35 31];
fprintf('Riego Mamdani (MATLAB): %.2f minutos\n', evalfis(fisM, entrada));
fprintf('Riego Sugeno  (MATLAB): %.2f minutos\n', evalfis(fisS, entrada));

% 5. Superficies de control 3D lado a lado
figure('Name', 'Validación Dual: Mamdani vs Sugeno', 'Color', 'w');
subplot(1,2,1); gensurf(fisM); title('Superficie Mamdani (Centroide)');
subplot(1,2,2); gensurf(fisS); title('Superficie Sugeno (Promedio Ponderado)');