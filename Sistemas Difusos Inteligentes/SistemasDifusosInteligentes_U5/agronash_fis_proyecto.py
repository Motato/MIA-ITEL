import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import skfuzzy as fuzz
from skfuzzy import control as ctrl

st.set_page_config(page_title="AgroNash-FIS | Proyecto Final Lógica Difusa", layout="wide")
st.title("AgroNash-FIS: Asignación Óptima de Agua Agrícola")
st.markdown("Sistema híbrido de **Inferencia Difusa Mamdani (27 Reglas)** y **Negociación Difusa de Nash**.")

# ==============================================================================
# CAPA 1: CONSTRUCCIÓN DEL SISTEMA MAMDANI (3 ENTRADAS -> 1 SALIDA)
# ==============================================================================
@st.cache_resource
def construir_agronash_fis():
    estres = ctrl.Antecedent(np.arange(0, 101, 1), 'Estres')
    disp   = ctrl.Antecedent(np.arange(0, 101, 1), 'Disponibilidad')
    efic   = ctrl.Antecedent(np.arange(0, 101, 1), 'Eficiencia')
    cuota  = ctrl.Consequent(np.arange(0, 101, 1), 'Cuota')

    estres['Bajo']     = fuzz.trapmf(estres.universe, [0, 0, 20, 40])
    estres['Moderado'] = fuzz.trimf(estres.universe,  [25, 50, 75])
    estres['Severo']   = fuzz.trapmf(estres.universe, [60, 80, 100, 100])

    disp['Critica'] = fuzz.trapmf(disp.universe, [0, 0, 20, 40])
    disp['Media']   = fuzz.trimf(disp.universe,  [25, 50, 75])
    disp['Alta']    = fuzz.trapmf(disp.universe, [60, 80, 100, 100])

    efic['Baja']  = fuzz.trapmf(efic.universe, [0, 0, 30, 50])
    efic['Media'] = fuzz.trimf(efic.universe,  [40, 60, 80])
    efic['Alta']  = fuzz.trapmf(efic.universe, [70, 85, 100, 100])

    cuota['Minima']   = fuzz.trimf(cuota.universe, [0, 0, 35])
    cuota['Moderada'] = fuzz.trimf(cuota.universe, [20, 50, 80])
    cuota['Optima']   = fuzz.trimf(cuota.universe, [65, 100, 100])

    # Generación sistemática de las 27 reglas exhaustivas (3x3x3)
    reglas = []
    for e in ['Bajo', 'Moderado', 'Severo']:
        for d in ['Critica', 'Media', 'Alta']:
            for f in ['Baja', 'Media', 'Alta']:
                if e == 'Bajo':
                    salida = 'Minima'
                elif e == 'Moderado':
                    if d == 'Critica':
                        salida = 'Minima'
                    elif d == 'Alta' and f == 'Alta':
                        salida = 'Optima'
                    else:
                        salida = 'Moderada'
                else:  # e == 'Severo'
                    if d == 'Critica' and f == 'Baja':
                        salida = 'Minima'
                    elif d == 'Critica':
                        salida = 'Moderada'
                    elif d == 'Media' and f == 'Baja':
                        salida = 'Moderada'
                    else:
                        salida = 'Optima'
                reglas.append(ctrl.Rule(estres[e] & disp[d] & efic[f], cuota[salida]))

    return ctrl.ControlSystem(reglas)

sistema_fis = construir_agronash_fis()

def evaluar_fis(e_val, d_val, f_val):
    sim = ctrl.ControlSystemSimulation(sistema_fis)
    sim.input['Estres'] = e_val
    sim.input['Disponibilidad'] = d_val
    sim.input['Eficiencia'] = f_val
    sim.compute()
    return float(sim.output['Cuota'])

# ==============================================================================
# PESTAÑAS INTERACTIVAS DEL PROYECTO
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "1. Simulador Mamdani (3 Entradas)",
    "2. Innovación: Negociación Difusa de Nash",
    "3. Validación Automática de Casos"
])

# --- PESTAÑA 1: MOTOR MAMDANI ---
with tab1:
    st.header("Capa 1: Evaluación Técnica de Parcela (Mamdani)")
    c1, c2 = st.columns([1, 2])
    with c1:
        e_in = st.slider("Estrés Hídrico del Cultivo (%):", 0, 100, 85)
        d_in = st.slider("Disponibilidad del Acuífero (%):", 0, 100, 55)
        f_in = st.slider("Eficiencia del Sistema de Riego (%):", 0, 100, 90)
        cuota_out = evaluar_fis(e_in, d_in, f_in)
        st.metric("Cuota de Asignación Recomendada", f"{cuota_out:.2f} %")

    with c2:
        # Superficie 3D fijando la eficiencia seleccionada
        eg = np.linspace(0, 100, 18)
        dg = np.linspace(0, 100, 18)
        E_m, D_m = np.meshgrid(eg, dg)
        Z_m = np.array([[evaluar_fis(E_m[i, j], D_m[i, j], f_in) for j in range(18)] for i in range(18)])

        fig3d = go.Figure(data=[
            go.Surface(z=Z_m, x=E_m, y=D_m, colorscale="YlGnBu", opacity=0.9),
            go.Scatter3d(x=[e_in], y=[d_in], z=[cuota_out], mode="markers", marker=dict(size=8, color="red"))
        ])
        fig3d.update_layout(
            title=f"Superficie de Decisión (Eficiencia fija en {f_in}%)",
            scene=dict(xaxis_title="Estrés (%)", yaxis_title="Acuífero (%)", zaxis_title="Cuota (%)"),
            height=450, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig3d, use_container_width=True)

# --- PESTAÑA 2: NEGOCIACIÓN DIFUSA DE NASH ---
with tab2:
    st.header("Capa 2 (Innovación): Resolución de Conflictos con Negociación Difusa de Nash")
    st.write("Reparto óptimo de un presupuesto hídrico limitado $B$ entre dos zonas agrícolas competidoras.")

    col_p1, col_p2, col_p3 = st.columns(3)
    B_total = col_p1.slider("Agua Total Disponible B (m³):", 500, 2000, 1000, 50)
    d1 = col_p2.number_input("Zona A - Mínimo Supervivencia d1 (m³):", 50, 800, 200)
    i1 = col_p2.number_input("Zona A - Demanda Ideal (m³):", 200, 1500, 700)
    d2 = col_p3.number_input("Zona B - Mínimo Supervivencia d2 (m³):", 50, 800, 150)
    i2 = col_p3.number_input("Zona B - Demanda Ideal (m³):", 200, 1500, 600)

    x1 = np.linspace(0, B_total, 1001)
    x2 = B_total - x1

    # Funciones de membresía de satisfacción lineal entre desacuerdo (0) e ideal (1)
    mu1 = np.clip((x1 - d1) / max(i1 - d1, 1e-6), 0, 1)
    mu2 = np.clip((x2 - d2) / max(i2 - d2, 1e-6), 0, 1)

    nash_prod = mu1 * mu2                # T-norma Producto (Fuzzy Nash)
    bz_min    = np.minimum(mu1, mu2)     # T-norma Mínimo (Bellman-Zadeh)

    idx_nash = int(np.argmax(nash_prod))
    idx_bz   = int(np.argmax(bz_min))

    m1, m2 = st.columns(2)
    m1.metric("Acuerdo Fuzzy Nash (Producto)", f"A: {x1[idx_nash]:.0f} m³ | B: {x2[idx_nash]:.0f} m³",
              f"Satisfacción: A={mu1[idx_nash]:.2f}, B={mu2[idx_nash]:.2f}")
    m2.metric("Acuerdo Bellman-Zadeh (Mínimo)", f"A: {x1[idx_bz]:.0f} m³ | B: {x2[idx_bz]:.0f} m³",
              f"Satisfacción: A={mu1[idx_bz]:.2f}, B={mu2[idx_bz]:.2f}")

    fig_n = go.Figure()
    fig_n.add_trace(go.Scatter(x=x1, y=mu1, name="Satisfacción Zona A (μ1)", line=dict(dash="dot")))
    fig_n.add_trace(go.Scatter(x=x1, y=mu2, name="Satisfacción Zona B (μ2)", line=dict(dash="dot")))
    fig_n.add_trace(go.Scatter(x=x1, y=nash_prod, name="Fuzzy Nash (μ1 · μ2)", line=dict(color="green", width=3)))
    fig_n.add_trace(go.Scatter(x=x1, y=bz_min, name="Bellman-Zadeh min(μ1, μ2)", line=dict(color="purple", width=3)))
    fig_n.update_layout(
        title="Comparativa de Negociación Difusa (Asignación a Zona A en eje X)",
        xaxis_title="Agua Asignada a Zona A (m³)", yaxis_title="Grado de Satisfacción / Decisión",
        template="plotly_white", height=400
    )
    st.plotly_chart(fig_n, use_container_width=True)

# --- PESTAÑA 3: VALIDACIÓN ---
with tab3:
    st.header("Validación Automatizada de Casos de Prueba")
    casos = [
        ("1. Sequía extrema + Riego ineficiente", 85, 15, 30, "Mínima (<30%)"),
        ("2. Estrés severo + Riego por goteo",    85, 55, 90, "Óptima (>75%)"),
        ("3. Cultivo sano + Acuífero lleno",      15, 85, 70, "Mínima (<30%)"),
        ("4. Condición promedio de temporada",    50, 50, 60, "Moderada (~50%)"),
    ]
    resultados = []
    for nom, e, d, f, esp in casos:
        val = evaluar_fis(e, d, f)
        resultados.append({"Escenario": nom, "E (%)": e, "D (%)": d, "F (%)": f,
                           "Esperado": esp, "Obtenido (%)": round(val, 2)})
    st.dataframe(pd.DataFrame(resultados), use_container_width=True, hide_index=True)