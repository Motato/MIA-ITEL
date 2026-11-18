# ============================================================
# Sistema Difuso para el Cálculo de Propinas
# ============================================================

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ============================================================
# 1. Definición de Universos de Discurso
# ============================================================
# Calidad de la comida: 0-10
# Calidad del servicio: 0-10
# Propina sugerida: 0-25 %
# ============================================================

comida = ctrl.Antecedent(np.arange(0, 11, 1), "comida")
servicio = ctrl.Antecedent(np.arange(0, 11, 1), "servicio")
propina = ctrl.Consequent(np.arange(0, 26, 1), "propina")

# ============================================================
# 2. Funciones de Membresía
# ============================================================

# Comida
comida["mala"] = fuzz.trimf(comida.universe, [0, 0, 5])
comida["regular"] = fuzz.trimf(comida.universe, [0, 5, 10])
comida["excelente"] = fuzz.trimf(comida.universe, [5, 10, 10])

# Servicio
servicio["malo"] = fuzz.trimf(servicio.universe, [0, 0, 5])
servicio["regular"] = fuzz.trimf(servicio.universe, [0, 5, 10])
servicio["excelente"] = fuzz.trimf(servicio.universe, [5, 10, 10])

# Propina
propina["baja"] = fuzz.trimf(propina.universe, [0, 0, 13])
propina["media"] = fuzz.trimf(propina.universe, [0, 13, 25])
propina["alta"] = fuzz.trimf(propina.universe, [13, 25, 25])

# ============================================================
# 3. Reglas Difusas
# ============================================================

# Si la comida O el servicio son malos,
# entonces la propina debe ser baja.
regla1 = ctrl.Rule(
    comida["mala"] | servicio["malo"],
    propina["baja"]
)

# Si el servicio es regular,
# entonces la propina debe ser media.
regla2 = ctrl.Rule(
    servicio["regular"],
    propina["media"]
)

# Si la comida O el servicio son excelentes,
# entonces la propina debe ser alta.
regla3 = ctrl.Rule(
    comida["excelente"] | servicio["excelente"],
    propina["alta"]
)

# ============================================================
# 4. Construcción del Sistema Difuso
# ============================================================

sistema_propinas = ctrl.ControlSystem(
    [regla1, regla2, regla3]
)

simulador = ctrl.ControlSystemSimulation(
    sistema_propinas
)

# ============================================================
# 5. Entradas del Sistema
# ============================================================

valor_comida = 6.5
valor_servicio = 9.8

simulador.input["comida"] = valor_comida
simulador.input["servicio"] = valor_servicio

# ============================================================
# 6. Inferencia Difusa y Desfuzificación
# ============================================================

simulador.compute()

propina_sugerida = simulador.output["propina"]

# ============================================================
# 7. Resultados
# ============================================================

print("=" * 50)
print("SISTEMA DIFUSO DE PROPINAS")
print("=" * 50)
print(f"Calidad de la comida   : {valor_comida}/10")
print(f"Calidad del servicio   : {valor_servicio}/10")
print(f"Propina recomendada    : {propina_sugerida:.2f}%")
print("=" * 50)

# ============================================================
# 8. Visualización
# ============================================================

# Funciones de membresía
comida.view()
servicio.view()

# Resultado final de propina
propina.view(sim=simulador)

plt.show()