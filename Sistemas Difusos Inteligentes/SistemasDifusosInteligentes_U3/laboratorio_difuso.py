import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Configuración Global de la Interfaz ---
st.set_page_config(page_title="Laboratorio de Lógica Difusa - Top 1", layout="wide")
st.title("Laboratorio Interactivo: Relaciones, Granularidad e Inferencia Difusa")
st.markdown("Simulador académico para el análisis de composiciones difusas, partición lingüística y control Mamdani.")

# ==============================================================================
# FUNCIONES VECTORIZADAS PARA COMPOSICIÓN DE RELACIONES (ACTIVIDAD PRINCIPAL)
# ==============================================================================
def composicion_max_min(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Composición Max-Min usando broadcasting 3D: s-norma=max, t-norma=min."""
    return np.max(np.minimum(R[:, :, None], S[None, :, :]), axis=1)

def composicion_max_prod(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Composición Max-Producto: s-norma=max, t-norma=producto algebraico."""
    return np.max(R[:, :, None] * S[None, :, :], axis=1)

def composicion_sumaprob_prod(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Composición Suma Probabilística - Producto: cambia s-norma y t-norma."""
    prod = R[:, :, None] * S[None, :, :]
    return 1.0 - np.prod(1.0 - prod, axis=1)

# ==============================================================================
# CONSTRUCCIÓN Y CACHÉ DEL SISTEMA MAMDANI (INCISO B)
# ==============================================================================
@st.cache_resource
def construir_fis_mamdani():
    temp = ctrl.Antecedent(np.arange(0, 101, 1), 'Temperatura')
    cpu  = ctrl.Antecedent(np.arange(0, 101, 1), 'Carga_CPU')
    vel  = ctrl.Consequent(np.arange(0, 101, 1), 'Velocidad')

    # Funciones de membresía
    temp['Baja']   = fuzz.trapmf(temp.universe, [0, 0, 25, 45])
    temp['Optima'] = fuzz.trimf(temp.universe,  [30, 50, 70])
    temp['Alta']   = fuzz.trapmf(temp.universe, [55, 75, 100, 100])

    cpu['Baja']  = fuzz.trapmf(cpu.universe, [0, 0, 25, 45])
    cpu['Media'] = fuzz.trimf(cpu.universe,  [30, 50, 70])
    cpu['Alta']  = fuzz.trapmf(cpu.universe, [55, 75, 100, 100])

    vel['Baja']  = fuzz.trimf(vel.universe, [0, 0, 45])
    vel['Media'] = fuzz.trimf(vel.universe, [25, 50, 75])
    vel['Alta']  = fuzz.trimf(vel.universe, [55, 100, 100])

    # Base de reglas exhaustiva (3x3 = 9 combinaciones)
    reglas = [
        ctrl.Rule(temp['Baja']   & cpu['Baja'],  vel['Baja']),
        ctrl.Rule(temp['Baja']   & cpu['Media'], vel['Baja']),
        ctrl.Rule(temp['Baja']   & cpu['Alta'],  vel['Media']),
        ctrl.Rule(temp['Optima'] & cpu['Baja'],  vel['Baja']),
        ctrl.Rule(temp['Optima'] & cpu['Media'], vel['Media']),
        ctrl.Rule(temp['Optima'] & cpu['Alta'],  vel['Alta']),
        ctrl.Rule(temp['Alta']   & cpu['Baja'],  vel['Media']),
        ctrl.Rule(temp['Alta']   & cpu['Media'], vel['Alta']),
        ctrl.Rule(temp['Alta']   & cpu['Alta'],  vel['Alta'])
    ]
    return ctrl.ControlSystem(reglas)

@st.cache_data
def calcular_superficie_3d(_sistema_ctrl):
    sim = ctrl.ControlSystemSimulation(_sistema_ctrl)
    t_vals = np.linspace(0, 100, 25)
    c_vals = np.linspace(0, 100, 25)
    T, C = np.meshgrid(t_vals, c_vals)
    Z = np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            sim.input['Temperatura'] = T[i, j]
            sim.input['Carga_CPU'] = C[i, j]
            sim.compute()
            Z[i, j] = sim.output['Velocidad']
    return T, C, Z

# ==============================================================================
# PESTAÑAS INTERACTIVAS DE LA APLICACIÓN
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "1. Relaciones y Composiciones",
    "2. Granularidad (3, 5 y 7 Casos)",
    "3. Sistema de Reglas y Superficie 3D"
])

# --- PESTAÑA 1: RELACIONES DIFUSAS ---
with tab1:
    st.header("Actividad Principal: Composición de Relaciones Difusas")
    st.write("Edite cualquier celda de las matrices $R$ y $S$ (valores entre 0 y 1) para recalcular en tiempo real las tres composiciones.")

    c1, c2 = st.columns(2)
    df_R_init = pd.DataFrame(
        [[0.90, 0.20, 0.00], [0.30, 0.80, 0.20], [0.10, 0.50, 0.90]],
        index=["CPU Baja", "CPU Media", "CPU Alta"],
        columns=["Temp Normal", "Temp Elevada", "Temp Crítica"]
    )
    df_S_init = pd.DataFrame(
        [[0.90, 0.10, 0.00], [0.20, 0.70, 0.30], [0.00, 0.40, 0.95]],
        index=["Temp Normal", "Temp Elevada", "Temp Crítica"],
        columns=["Riesgo Bajo", "Riesgo Moderado", "Riesgo Inminente"]
    )

    with c1:
        st.subheader("Matriz R (Carga CPU → Temperatura)")
        df_R = st.data_editor(df_R_init, key="mat_R")
    with c2:
        st.subheader("Matriz S (Temperatura → Riesgo Fallo)")
        df_S = st.data_editor(df_S_init, key="mat_S")

    R_np, S_np = np.clip(df_R.to_numpy(dtype=float), 0, 1), np.clip(df_S.to_numpy(dtype=float), 0, 1)

    T_mm = composicion_max_min(R_np, S_np)
    T_mp = composicion_max_prod(R_np, S_np)
    T_sp = composicion_sumaprob_prod(R_np, S_np)

    st.subheader("Resultados Comparativos ($T = R \\circ S$)")
    col_a, col_b, col_c = st.columns(3)
    nombres_x = ["CPU Baja", "CPU Media", "CPU Alta"]
    nombres_z = ["Riesgo Bajo", "Riesgo Mod.", "Riesgo Inm."]

    for col, mat, titulo in [
        (col_a, T_mm, "Max-Min (Estándar)"),
        (col_b, T_mp, "Max-Producto"),
        (col_c, T_sp, "Suma Prob. - Producto")
    ]:
        with col:
            fig = px.imshow(
                np.round(mat, 3), text_auto=True, x=nombres_z, y=nombres_x,
                color_continuous_scale="Viridis", zmin=0, zmax=1, title=titulo
            )
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

# --- PESTAÑA 2: GRANULARIDAD LINGÜÍSTICA ---
with tab2:
    st.header("Inciso a): Granularidad de Valores Lingüísticos")
    x_univ = np.linspace(0, 100, 500)

    col_ctrl, col_viz = st.columns([1, 2.5])
    with col_ctrl:
        caso = st.radio(
            "Seleccione el caso de estudio:",
            ["Caso 1: 3 Valores (Trap/Tri)", "Caso 2: 5 Valores (Gaussianas)", "Caso 3: 7 Valores (Campana Gen.)"]
        )
        x_eval = st.slider("Evaluar valor exacto de Temperatura (°C):", 0.0, 100.0, 42.0, 0.5)
        alpha = st.slider("Nivel de Corte α (Alpha-cut):", 0.0, 1.0, 0.5, 0.05)

    conjuntos = {}
    if "3 Valores" in caso:
        conjuntos["Baja"]   = fuzz.trapmf(x_univ, [0, 0, 25, 45])
        conjuntos["Óptima"] = fuzz.trimf(x_univ, [30, 50, 70])
        conjuntos["Alta"]   = fuzz.trapmf(x_univ, [55, 75, 100, 100])
    elif "5 Valores" in caso:
        nombres5 = ["Muy Baja", "Baja", "Óptima", "Alta", "Crítica"]
        for nom, cen in zip(nombres5, [10, 30, 50, 70, 90]):
            conjuntos[nom] = fuzz.gaussmf(x_univ, cen, 10)
    else:
        nombres7 = ["Ext. Baja", "Muy Baja", "Baja", "Óptima", "Alta", "Muy Alta", "Crítica"]
        for nom, cen in zip(nombres7, np.linspace(5, 95, 7)):
            conjuntos[nom] = fuzz.gbellmf(x_univ, 8, 2.5, cen)

    with col_viz:
        fig_g = go.Figure()
        evaluaciones = {}
        for nombre, y_vals in conjuntos.items():
            fig_g.add_trace(go.Scatter(x=x_univ, y=y_vals, mode='lines', name=nombre, line=dict(width=2.5)))
            evaluaciones[nombre] = float(np.interp(x_eval, x_univ, y_vals))

        # Línea vertical del punto evaluado y horizontal del corte alfa
        fig_g.add_vline(x=x_eval, line_dash="dash", line_color="black", annotation_text=f"x = {x_eval}°C")
        fig_g.add_hline(y=alpha, line_dash="dot", line_color="red", annotation_text=f"Corte α = {alpha}")
        fig_g.update_layout(
            title=f"Funciones de Membresía — {caso}",
            xaxis_title="Temperatura (°C)", yaxis_title="Grado de Membresía μ(x)",
            yaxis=dict(range=[-0.05, 1.1]), template="plotly_white", height=420
        )
        st.plotly_chart(fig_g, use_container_width=True)

    st.subheader(f"Grados de Membresía Activos para x = {x_eval}°C")
    df_eval = pd.DataFrame([
        {"Conjunto Difuso": k, "Membresía μ(x)": round(v, 4), f"¿Supera Corte α ({alpha})?": "Sí" if v >= alpha else "No"}
        for k, v in evaluaciones.items()
    ])
    st.dataframe(df_eval, use_container_width=True, hide_index=True)

# --- PESTAÑA 3: SISTEMA DE REGLAS Y SUPERFICIE 3D ---
with tab3:
    st.header("Inciso b): Controlador Mamdani Exhaustivo (9 Reglas)")
    sistema = construir_fis_mamdani()
    sim = ctrl.ControlSystemSimulation(sistema)

    c_in, c_out = st.columns([1, 2])
    with c_in:
        st.subheader("Entradas del Sistema")
        t_in = st.slider("Temperatura Actual (°C)", 0, 100, 68)
        c_in_val = st.slider("Carga de CPU Actual (%)", 0, 100, 82)

        sim.input['Temperatura'] = t_in
        sim.input['Carga_CPU'] = c_in_val
        sim.compute()
        vel_out = sim.output['Velocidad']

        st.metric(label="Velocidad Recomendada del Ventilador (Centroide)", value=f"{vel_out:.2f} %")
        st.info("El sistema evalúa simultáneamente las 9 combinaciones de la Matriz FAM y defusifica mediante el método del centroide.")

    with c_out:
        T_grid, C_grid, Z_grid = calcular_superficie_3d(sistema)
        fig_3d = go.Figure(data=[
            go.Surface(z=Z_grid, x=T_grid, y=C_grid, colorscale='Turbo', opacity=0.9),
            go.Scatter3d(
                x=[t_in], y=[c_in_val], z=[vel_out], mode='markers',
                marker=dict(size=8, color='black'), name='Punto Actual'
            )
        ])
        fig_3d.update_layout(
            title="Superficie de Control 3D (Temperatura × Carga CPU → Velocidad)",
            scene=dict(xaxis_title='Temperatura (°C)', yaxis_title='Carga CPU (%)', zaxis_title='Velocidad (%)'),
            height=500, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        