import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import skfuzzy as fuzz

st.set_page_config(page_title="Laboratorio EV: Tsukamoto, Mamdani y Sugeno", layout="wide")
st.title("Laboratorio Interactivo: Carga Rápida de Baterías EV")
st.markdown("Comparativa en tiempo real de **Tsukamoto (Monótono)**, **Mamdani (Centroide)** y **Sugeno (1.er Orden)** con partición en rejilla.")

# ==============================================================================
# UNIVERSOS DE DISCURSO Y FUNCIONES DE MEMBRESÍA
# ==============================================================================
x_soc  = np.linspace(0, 100, 201)   # Estado de Carga (%)
y_temp = np.linspace(15, 65, 201)   # Temperatura (°C)
z_curr = np.linspace(0, 100, 401)   # Corriente de Carga (A)

s_bajo  = fuzz.trapmf(x_soc, [0, 0, 20, 40])
s_medio = fuzz.trimf(x_soc,  [25, 50, 75])
s_alto  = fuzz.trapmf(x_soc, [60, 80, 100, 100])

t_norm = fuzz.trapmf(y_temp, [15, 15, 25, 35])
t_cal  = fuzz.trimf(y_temp,  [30, 40, 50])
t_crit = fuzz.trapmf(y_temp, [45, 55, 65, 65])

c_baja  = fuzz.trimf(z_curr, [0, 0, 40])
c_media = fuzz.trimf(z_curr, [25, 50, 75])
c_alta  = fuzz.trimf(z_curr, [60, 100, 100])

FAM = [
    ("R1", "Bajo",  "Norm", "Alta"),  ("R2", "Bajo",  "Cal", "Media"), ("R3", "Bajo",  "Crit", "Baja"),
    ("R4", "Medio", "Norm", "Media"), ("R5", "Medio", "Cal", "Media"), ("R6", "Medio", "Crit", "Baja"),
    ("R7", "Alto",  "Norm", "Baja"),  ("R8", "Alto",  "Cal", "Baja"),  ("R9", "Alto",  "Crit", "Baja"),
]

# ==============================================================================
# MOTORES DE INFERENCIA (SISO Y MISO)
# ==============================================================================
def siso_modelos(soc_val: float):
    a1 = float(fuzz.interp_membership(x_soc, s_bajo,  soc_val))  # -> Alta
    a2 = float(fuzz.interp_membership(x_soc, s_medio, soc_val))  # -> Media
    a3 = float(fuzz.interp_membership(x_soc, s_alto,  soc_val))  # -> Baja
    den = max(a1 + a2 + a3, 1e-9)

    # Tsukamoto SISO (Inversión monótona)
    out_t = (a1 * (50.0 + 50.0 * a1) + a2 * (20.0 + 50.0 * a2) + a3 * (40.0 * (1.0 - a3))) / den

    # Mamdani SISO (Centroide)
    agg = np.fmax(np.minimum(a1, c_alta), np.fmax(np.minimum(a2, c_media), np.minimum(a3, c_baja)))
    out_m = float(fuzz.defuzz(z_curr, agg, 'centroid'))

    # Sugeno SISO (1.er Orden)
    z1 = np.clip(-0.4 * soc_val + 95, 0, 100)
    z2 = np.clip(-0.3 * soc_val + 65, 0, 100)
    z3 = np.clip(-0.2 * soc_val + 30, 0, 100)
    out_s = (a1 * z1 + a2 * z2 + a3 * z3) / den

    return out_t, out_m, out_s, (a1, a2, a3)

def evaluar_fam(soc_val: float, temp_val: float):
    ms = {
        "Bajo":  float(fuzz.interp_membership(x_soc, s_bajo,  soc_val)),
        "Medio": float(fuzz.interp_membership(x_soc, s_medio, soc_val)),
        "Alto":  float(fuzz.interp_membership(x_soc, s_alto,  soc_val)),
    }
    mt = {
        "Norm": float(fuzz.interp_membership(y_temp, t_norm, temp_val)),
        "Cal":  float(fuzz.interp_membership(y_temp, t_cal,  temp_val)),
        "Crit": float(fuzz.interp_membership(y_temp, t_crit, temp_val)),
    }
    return [(r, s, t, cons, min(ms[s], mt[t])) for r, s, t, cons in FAM]

def miso_tsukamoto(soc_val: float, temp_val: float) -> float:
    num, den = 0.0, 0.0
    for _, _, _, cons, a in evaluar_fam(soc_val, temp_val):
        if a > 0:
            if cons == "Baja":    zi = 40.0 * (1.0 - a)
            elif cons == "Media": zi = 20.0 + 50.0 * a
            else:                 zi = 50.0 + 50.0 * a
            num += a * zi; den += a
    return num / max(den, 1e-9)

def miso_mamdani(soc_val: float, temp_val: float) -> float:
    agg = np.zeros_like(z_curr)
    mapa = {"Baja": c_baja, "Media": c_media, "Alta": c_alta}
    for _, _, _, cons, a in evaluar_fam(soc_val, temp_val):
        if a > 0:
            agg = np.fmax(agg, np.minimum(a, mapa[cons]))
    return float(fuzz.defuzz(z_curr, agg, 'centroid')) if np.sum(agg) > 0 else 0.0

def miso_sugeno(soc_val: float, temp_val: float) -> float:
    num, den = 0.0, 0.0
    for _, _, _, cons, a in evaluar_fam(soc_val, temp_val):
        if a > 0:
            if cons == "Baja":    zi = np.clip(-0.2 * soc_val - 0.3 * temp_val + 35,  0, 100)
            elif cons == "Media": zi = np.clip(-0.3 * soc_val - 0.5 * temp_val + 80,  0, 100)
            else:                 zi = np.clip(-0.4 * soc_val - 0.6 * temp_val + 115, 0, 100)
            num += a * zi; den += a
    return num / max(den, 1e-9)

@st.cache_data
def precalcular_mallas():
    s_1d = np.linspace(0, 100, 101)
    t_1d, m_1d, sg_1d = zip(*[siso_modelos(s)[:3] for s in s_1d])

    sg = np.linspace(0, 100, 22)
    tg = np.linspace(15, 65, 22)
    S, T = np.meshgrid(sg, tg)
    ZT, ZM, ZS = np.zeros_like(S), np.zeros_like(S), np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            ZT[i, j] = miso_tsukamoto(S[i, j], T[i, j])
            ZM[i, j] = miso_mamdani(S[i, j], T[i, j])
            ZS[i, j] = miso_sugeno(S[i, j], T[i, j])
    return s_1d, np.array(t_1d), np.array(m_1d), np.array(sg_1d), S, T, ZT, ZM, ZS

s_1d, c_t1d, c_m1d, c_s1d, S_g, T_g, ZT, ZM, ZS = precalcular_mallas()

# ==============================================================================
# INTERFAZ POR PESTAÑAS
# ==============================================================================
tab1, tab2 = st.tabs([
    "1. SISO (1 Entrada): Tsukamoto vs. Mamdani y Sugeno",
    "2. MISO (2 Entradas): Superficies 3D, Benchmark y Reglas"
])

with tab1:
    st.header("Modelo SISO: Estado de Carga (SoC) → Corriente de Carga (A)")
    c1, c2 = st.columns([1, 2.2])
    with c1:
        soc_in1 = st.slider("Estado de Carga SoC (%):", 0.0, 100.0, 30.0, 1.0, key="s1")
        ot1, om1, os1, (a1, a2, a3) = siso_modelos(soc_in1)
        st.metric("Tsukamoto 1D (Monótono)", f"{ot1:.2f} A")
        st.metric("Mamdani 1D (Centroide)",  f"{om1:.2f} A")
        st.metric("Sugeno 1D (1.er Orden)",  f"{os1:.2f} A")
        st.info(f"**Membresías SoC:** Bajo={a1:.2f} | Medio={a2:.2f} | Alto={a3:.2f}")

    with c2:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=s_1d, y=c_t1d, name="Tsukamoto (Monótono)", line=dict(color="#2ca02c", width=3)))
        fig1.add_trace(go.Scatter(x=s_1d, y=c_m1d, name="Mamdani (Centroide)",  line=dict(color="#1f77b4", width=2.5, dash="dash")))
        fig1.add_trace(go.Scatter(x=s_1d, y=c_s1d, name="Sugeno (1.er Orden)",  line=dict(color="#ff7f0e", width=2.5, dash="dot")))
        fig1.add_trace(go.Scatter(x=[soc_in1]*3, y=[ot1, om1, os1], mode="markers", name="Punto Actual", marker=dict(size=9, color="red")))
        fig1.update_layout(
            title="Curvas de Transferencia SISO (1 Entrada → 1 Salida)",
            xaxis_title="Estado de Carga SoC (%)", yaxis_title="Corriente de Carga (A)",
            template="plotly_white", height=420
        )
        st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.header("Modelo MISO: SoC y Temperatura → Corriente de Carga (A)")
    cs1, cs2 = st.columns(2)
    soc_in2  = cs1.slider("Estado de Carga SoC (%):", 0.0, 100.0, 30.0, 1.0, key="s2")
    temp_in2 = cs2.slider("Temperatura de Batería (°C):", 15.0, 65.0, 42.0, 0.5, key="t2")

    t0 = time.perf_counter(); out_t = miso_tsukamoto(soc_in2, temp_in2); tt = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); out_m = miso_mamdani(soc_in2, temp_in2);   tm = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); out_s = miso_sugeno(soc_in2, temp_in2);    ts = (time.perf_counter() - t0) * 1000

    k1, k2, k3 = st.columns(3)
    k1.metric("Tsukamoto (Monótono)", f"{out_t:.2f} A", f"{tt:.3f} ms", delta_color="off")
    k2.metric("Mamdani (Centroide)",  f"{out_m:.2f} A", f"{tm:.3f} ms", delta_color="off")
    k3.metric("Sugeno (1.er Orden)",  f"{out_s:.2f} A", f"{ts:.3f} ms", delta_color="off")

    g1, g2, g3 = st.columns(3)
    for col, Z_mat, titulo, pt in [
        (g1, ZT, "Superficie Tsukamoto",    out_t),
        (g2, ZM, "Superficie Mamdani",      out_m),
        (g3, ZS, "Superficie Sugeno (TSK)", out_s),
    ]:
        with col:
            fig3 = go.Figure(data=[
                go.Surface(z=Z_mat, x=S_g, y=T_g, colorscale="Viridis", showscale=False),
                go.Scatter3d(x=[soc_in2], y=[temp_in2], z=[pt], mode="markers", marker=dict(size=7, color="red"))
            ])
            fig3.update_layout(
                title=titulo, height=360, margin=dict(l=0, r=0, t=35, b=0),
                scene=dict(xaxis_title="SoC (%)", yaxis_title="Temp (°C)", zaxis_title="Corr (A)")
            )
            st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Auditoría de Reglas Activas (Partición en Rejilla 3×3)")
    fam_eval = evaluar_fam(soc_in2, temp_in2)
    df_fam = pd.DataFrame([
        {"Regla": r, "Antecedentes (AND)": f"{s} ∧ {t}", "Consecuente": c, "Fuerza de Disparo (α)": round(a, 3)}
        for r, s, t, c, a in fam_eval
    ])
    st.dataframe(df_fam, use_container_width=True, hide_index=True)
    
