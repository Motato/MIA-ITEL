import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==============================================================================
# CONFIGURACIÓN DE INTERFAZ
# ==============================================================================
st.set_page_config(page_title="Simulador de Inferencia Difusa - Top 1", layout="wide")
st.title("Laboratorio Interactivo: Reglas If-Then e Inferencia Composicional")
st.markdown("Simulación de **Razonamiento Difuso (CRI)**, **Granularidad Lingüística** y **Control Exhaustivo (Matriz FAM)** para un Invernadero Inteligente.")

# Universos de discurso globales
x_hum  = np.linspace(0, 100, 501)  # Humedad del suelo (%)
y_temp = np.linspace(0, 45, 451)   # Temperatura ambiente (°C)
z_rieg = np.linspace(0, 60, 601)   # Duración de riego (minutos)

# Funciones de membresía base (3 términos por variable)
hum_seca   = fuzz.trapmf(x_hum, [0, 0, 25, 45])
hum_optima = fuzz.trimf(x_hum,  [30, 50, 70])
hum_humeda = fuzz.trapmf(x_hum, [55, 75, 100, 100])

temp_baja     = fuzz.trapmf(y_temp, [0, 0, 12, 20])
temp_templada = fuzz.trimf(y_temp,  [14, 24, 34])
temp_alta     = fuzz.trapmf(y_temp, [26, 35, 45, 45])

rieg_corto = fuzz.trimf(z_rieg, [0, 0, 25])
rieg_medio = fuzz.trimf(z_rieg, [15, 30, 45])
rieg_largo = fuzz.trimf(z_rieg, [35, 60, 60])

def implicar(alpha: float, mf_consecuente: np.ndarray, metodo: str) -> np.ndarray:
    """Aplica implicación de Mamdani (mínimo) o Larsen (producto)."""
    if "Mamdani" in metodo:
        return np.minimum(alpha, mf_consecuente)
    return alpha * mf_consecuente

def defusificar_seguro(universo: np.ndarray, agregado: np.ndarray, metodo: str) -> float:
    """Evita división por cero si ninguna regla se activa."""
    if np.sum(agregado) == 0:
        return 0.0
    return float(fuzz.defuzz(universo, agregado, metodo))

# ==============================================================================
# CACHÉ DEL SISTEMA EXHAUSTIVO MAMDANI (EVALUACIÓN B)
# ==============================================================================
@st.cache_resource
def crear_sistema_exhaustivo():
    hum  = ctrl.Antecedent(np.arange(0, 101, 1), 'Humedad')
    temp = ctrl.Antecedent(np.arange(0, 46, 1), 'Temperatura')
    rieg = ctrl.Consequent(np.arange(0, 61, 1), 'Riego')

    hum['Seca']   = fuzz.trapmf(hum.universe, [0, 0, 25, 45])
    hum['Optima'] = fuzz.trimf(hum.universe,  [30, 50, 70])
    hum['Humeda'] = fuzz.trapmf(hum.universe, [55, 75, 100, 100])

    temp['Baja']     = fuzz.trapmf(temp.universe, [0, 0, 12, 20])
    temp['Templada'] = fuzz.trimf(temp.universe,  [14, 24, 34])
    temp['Alta']     = fuzz.trapmf(temp.universe, [26, 35, 45, 45])

    rieg['Corto'] = fuzz.trimf(rieg.universe, [0, 0, 25])
    rieg['Medio'] = fuzz.trimf(rieg.universe, [15, 30, 45])
    rieg['Largo'] = fuzz.trimf(rieg.universe, [35, 60, 60])

    # 9 Reglas Exhaustivas (Matriz FAM 3x3)
    reglas = [
        ctrl.Rule(hum['Seca']   & temp['Baja'],     rieg['Medio']),
        ctrl.Rule(hum['Seca']   & temp['Templada'], rieg['Largo']),
        ctrl.Rule(hum['Seca']   & temp['Alta'],     rieg['Largo']),
        ctrl.Rule(hum['Optima'] & temp['Baja'],     rieg['Corto']),
        ctrl.Rule(hum['Optima'] & temp['Templada'], rieg['Medio']),
        ctrl.Rule(hum['Optima'] & temp['Alta'],     rieg['Largo']),
        ctrl.Rule(hum['Humeda'] & temp['Baja'],     rieg['Corto']),
        ctrl.Rule(hum['Humeda'] & temp['Templada'], rieg['Corto']),
        ctrl.Rule(hum['Humeda'] & temp['Alta'],     rieg['Medio']),
    ]
    return ctrl.ControlSystem(reglas)

@st.cache_data
def calcular_superficie_3d(_sistema_ctrl):
    sim = ctrl.ControlSystemSimulation(_sistema_ctrl)
    h_vals = np.linspace(0, 100, 26)
    t_vals = np.linspace(0, 45, 26)
    H, T = np.meshgrid(h_vals, t_vals)
    Z = np.zeros_like(H)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            sim.input['Humedad'] = H[i, j]
            sim.input['Temperatura'] = T[i, j]
            sim.compute()
            Z[i, j] = sim.output['Riego']
    return H, T, Z

# ==============================================================================
# PESTAÑAS INTERACTIVAS
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "1. Inferencia Composicional (Casos a, b, c)",
    "2. Evaluación a): Granularidad (3, 5, 7)",
    "3. Evaluación b): Sistema Exhaustivo y 3D"
])

# --- PESTAÑA 1: INFERENCIA COMPOSICIONAL PASO A PASO ---
with tab1:
    st.header("Simulación de la Regla de Inferencia Composicional (CRI)")
    col_cfg, col_graf = st.columns([1, 2.2])

    with col_cfg:
        caso_cri = st.radio(
            "Seleccione el esquema de reglas:",
            [
                "Caso a) 1 Regla, 1 Antecedente",
                "Caso b) 1 Regla, Múltiples Antecedentes",
                "Caso c) Múltiples Reglas, Múltiples Antecedentes"
            ]
        )
        metodo_imp = st.selectbox("Operador de Implicación:", ["Mamdani (Recorte min)", "Larsen (Escalado producto)"])
        h_in = st.slider("Humedad del Suelo (%):", 0.0, 100.0, 35.0, 1.0, key="cri_h")
        t_in = st.slider("Temperatura Ambiente (°C):", 0.0, 45.0, 31.0, 0.5, key="cri_t")

        # Evaluar grados de membresía puntuales
        mu_h_seca   = float(fuzz.interp_membership(x_hum, hum_seca, h_in))
        mu_h_optima = float(fuzz.interp_membership(x_hum, hum_optima, h_in))
        mu_h_humeda = float(fuzz.interp_membership(x_hum, hum_humeda, h_in))

        mu_t_baja     = float(fuzz.interp_membership(y_temp, temp_baja, t_in))
        mu_t_templada = float(fuzz.interp_membership(y_temp, temp_templada, t_in))
        mu_t_alta     = float(fuzz.interp_membership(y_temp, temp_alta, t_in))

        if "Caso a)" in caso_cri:
            alpha1 = mu_h_seca
            salida_agregada = implicar(alpha1, rieg_largo, metodo_imp)
            st.info(f"**Regla:** SI Humedad es Seca ({mu_h_seca:.2f}) ENTONCES Riego es Largo.\n\n**Activación α₁ = {alpha1:.3f}**")
        elif "Caso b)" in caso_cri:
            alpha1 = min(mu_h_seca, mu_t_alta)
            salida_agregada = implicar(alpha1, rieg_largo, metodo_imp)
            st.info(f"**Regla:** SI Humedad es Seca ({mu_h_seca:.2f}) Y Temp. es Alta ({mu_t_alta:.2f}) ENTONCES Riego es Largo.\n\n**Activación α₁ = min = {alpha1:.3f}**")
        else:
            a1 = min(mu_h_seca, mu_t_alta)        # -> Largo
            a2 = min(mu_h_optima, mu_t_templada)  # -> Medio
            a3 = min(mu_h_humeda, mu_t_baja)      # -> Corto
            r1_out = implicar(a1, rieg_largo, metodo_imp)
            r2_out = implicar(a2, rieg_medio, metodo_imp)
            r3_out = implicar(a3, rieg_corto, metodo_imp)
            salida_agregada = np.fmax(r1_out, np.fmax(r2_out, r3_out))
            st.info(
                f"**Activación de Reglas (AND = min):**\n"
                f"- R₁ (Seca ∧ Alta → Largo): **α₁ = {a1:.3f}**\n"
                f"- R₂ (Óptima ∧ Templada → Medio): **α₂ = {a2:.3f}**\n"
                f"- R₃ (Húmeda ∧ Baja → Corto): **α₃ = {a3:.3f}**"
            )

        z_centroid = defusificar_seguro(z_rieg, salida_agregada, 'centroid')
        z_bisector = defusificar_seguro(z_rieg, salida_agregada, 'bisector')
        z_mom      = defusificar_seguro(z_rieg, salida_agregada, 'mom')

        m1, m2, m3 = st.columns(3)
        m1.metric("Centroide", f"{z_centroid:.1f} min")
        m2.metric("Bisector", f"{z_bisector:.1f} min")
        m3.metric("MOM", f"{z_mom:.1f} min")

    with col_graf:
        fig_cri = go.Figure()
        fig_cri.add_trace(go.Scatter(x=z_rieg, y=rieg_corto, name="Consecuente: Corto", line=dict(dash="dot", color="gray")))
        fig_cri.add_trace(go.Scatter(x=z_rieg, y=rieg_medio, name="Consecuente: Medio", line=dict(dash="dot", color="orange")))
        fig_cri.add_trace(go.Scatter(x=z_rieg, y=rieg_largo, name="Consecuente: Largo", line=dict(dash="dot", color="blue")))
        fig_cri.add_trace(go.Scatter(
            x=z_rieg, y=salida_agregada, name="Salida Inferida (Agregación)",
            fill="tozeroy", line=dict(color="#008080", width=3)
        ))
        if z_centroid > 0:
            fig_cri.add_vline(x=z_centroid, line_dash="dash", line_color="red", annotation_text=f"Centroide = {z_centroid:.2f} min")
        fig_cri.update_layout(
            title=f"Conjunto Difuso Resultante — {caso_cri} ({metodo_imp})",
            xaxis_title="Duración del Riego (minutos)", yaxis_title="Grado de Membresía μ(z)",
            yaxis=dict(range=[-0.05, 1.1]), template="plotly_white", height=460
        )
        st.plotly_chart(fig_cri, use_container_width=True)

# --- PESTAÑA 2: EVALUACIÓN A (GRANULARIDAD 3, 5, 7) ---
with tab2:
    st.header("Evaluación a): Comparación de Granularidad en Funciones de Membresía")
    c_izq, c_der = st.columns([1, 2.3])

    with c_izq:
        opcion_g = st.radio(
            "Número de valores difusos:",
            ["Caso 1: 3 Valores (Trapezoidal/Triangular)", "Caso 2: 5 Valores (Gaussianas)", "Caso 3: 7 Valores (Campana Generalizada)"]
        )
        h_eval = st.slider("Humedad de prueba (%):", 0.0, 100.0, 42.0, 0.5, key="gran_h")
        corte_a = st.slider("Umbral de Corte α:", 0.0, 1.0, 0.5, 0.05, key="gran_a")

    conjuntos = {}
    if "3 Valores" in opcion_g:
        conjuntos = {"Seco": hum_seca, "Óptimo": hum_optima, "Húmedo": hum_humeda}
    elif "5 Valores" in opcion_g:
        for nom, c in zip(["Muy Seco", "Seco", "Óptimo", "Húmedo", "Saturado"], [10, 30, 50, 70, 90]):
            conjuntos[nom] = fuzz.gaussmf(x_hum, c, 10)
    else:
        for nom, c in zip(["Árido", "Muy Seco", "Seco", "Óptimo", "Húmedo", "Muy Húmedo", "Saturado"], np.linspace(5, 95, 7)):
            conjuntos[nom] = fuzz.gbellmf(x_hum, 8, 2.5, c)

    with c_der:
        fig_g = go.Figure()
        filas = []
        for nom, yv in conjuntos.items():
            fig_g.add_trace(go.Scatter(x=x_hum, y=yv, mode="lines", name=nom, line=dict(width=2.5)))
            val = float(fuzz.interp_membership(x_hum, yv, h_eval))
            filas.append({"Valor Lingüístico": nom, "Membresía μ(x)": round(val, 4), f"Pertenece al Corte α ({corte_a})": "Sí" if val >= corte_a else "No"})

        fig_g.add_vline(x=h_eval, line_dash="dash", line_color="black", annotation_text=f"x = {h_eval}%")
        fig_g.add_hline(y=corte_a, line_dash="dot", line_color="red", annotation_text=f"α = {corte_a}")
        fig_g.update_layout(
            title=f"Variable: Humedad del Suelo — {opcion_g}",
            xaxis_title="Humedad del Suelo (%)", yaxis_title="Grado de Membresía μ(x)",
            yaxis=dict(range=[-0.05, 1.1]), template="plotly_white", height=400
        )
        st.plotly_chart(fig_g, use_container_width=True)

    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

# --- PESTAÑA 3: EVALUACIÓN B (SISTEMA EXHAUSTIVO 9 REGLAS + 3D) ---
with tab3:
    st.header("Evaluación b): Controlador Exhaustivo (Matriz FAM 3×3 = 9 Reglas)")
    sistema_ex = crear_sistema_exhaustivo()
    sim = ctrl.ControlSystemSimulation(sistema_ex)

    col_in, col_3d = st.columns([1, 2])
    with col_in:
        h_sys = st.slider("Humedad del Suelo (%):", 0, 100, 35, key="sys_h")
        t_sys = st.slider("Temperatura Ambiente (°C):", 0, 45, 31, key="sys_t")

        sim.input['Humedad'] = h_sys
        sim.input['Temperatura'] = t_sys
        sim.compute()
        r_out = sim.output['Riego']

        st.metric("Duración de Riego Óptima (Mamdani)", f"{r_out:.2f} minutos")

        # Tabla en vivo de disparo de las 9 reglas exhaustivas
        mh = {
            "Seca": float(fuzz.interp_membership(x_hum, hum_seca, h_sys)),
            "Óptima": float(fuzz.interp_membership(x_hum, hum_optima, h_sys)),
            "Húmeda": float(fuzz.interp_membership(x_hum, hum_humeda, h_sys)),
        }
        mt = {
            "Baja": float(fuzz.interp_membership(y_temp, temp_baja, t_sys)),
            "Templada": float(fuzz.interp_membership(y_temp, temp_templada, t_sys)),
            "Alta": float(fuzz.interp_membership(y_temp, temp_alta, t_sys)),
        }
        fam = [
            ("R1", "Seca", "Baja", "Medio"), ("R2", "Seca", "Templada", "Largo"), ("R3", "Seca", "Alta", "Largo"),
            ("R4", "Óptima", "Baja", "Corto"), ("R5", "Óptima", "Templada", "Medio"), ("R6", "Óptima", "Alta", "Largo"),
            ("R7", "Húmeda", "Baja", "Corto"), ("R8", "Húmeda", "Templada", "Corto"), ("R9", "Húmeda", "Alta", "Medio")
        ]
        df_disparo = pd.DataFrame([
            {"Regla": r, "Antecedentes (AND)": f"{h} ∧ {t}", "Consecuente": c, "Fuerza de Disparo (α)": round(min(mh[h], mt[t]), 3)}
            for r, h, t, c in fam
        ])
        st.subheader("Activación de Reglas en Vivo")
        st.dataframe(df_disparo, use_container_width=True, hide_index=True, height=350)

    with col_3d:
        H_g, T_g, Z_g = calcular_superficie_3d(sistema_ex)
        fig3d = go.Figure(data=[
            go.Surface(z=Z_g, x=H_g, y=T_g, colorscale="Viridis", opacity=0.92),
            go.Scatter3d(x=[h_sys], y=[t_sys], z=[r_out], mode="markers", marker=dict(size=8, color="red"), name="Operación Actual")
        ])
        fig3d.update_layout(
            title="Superficie de Control 3D (Humedad × Temperatura → Riego)",
            scene=dict(xaxis_title="Humedad (%)", yaxis_title="Temperatura (°C)", zaxis_title="Riego (min)"),
            height=540, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig3d, use_container_width=True)