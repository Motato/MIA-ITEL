import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import skfuzzy as fuzz

st.set_page_config(page_title="Laboratorio ACC: Sugeno, Mamdani y Tsukamoto", layout="wide")
st.title("Laboratorio Interactivo: Control de Crucero Adaptativo (ACC)")
st.markdown("Comparativa en tiempo real de los modelos **Sugeno (1.er orden)**, **Mamdani (Centroide)** y **Tsukamoto (Monótono)**.")

# ==============================================================================
# UNIVERSOS DE DISCURSO Y FUNCIONES DE MEMBRESÍA COMPARTIDAS
# ==============================================================================
x_dist = np.linspace(0, 100, 201)     # Distancia (m)
y_vrel = np.linspace(-30, 30, 201)    # Velocidad relativa (km/h)
z_ctrl = np.linspace(-100, 100, 401)  # Acción de control (%)

d_corta = fuzz.trapmf(x_dist, [0, 0, 20, 40])
d_media = fuzz.trimf(x_dist,  [25, 50, 75])
d_larga = fuzz.trapmf(x_dist, [60, 80, 100, 100])

v_alej = fuzz.trapmf(y_vrel, [-30, -30, -15, 0])
v_ig   = fuzz.trimf(y_vrel,  [-10, 0, 10])
v_acer = fuzz.trapmf(y_vrel, [0, 15, 30, 30])

c_fren = fuzz.trimf(z_ctrl, [-100, -100, -10])
c_mant = fuzz.trimf(z_ctrl, [-30, 0, 30])
c_acel = fuzz.trimf(z_ctrl, [10, 100, 100])

FAM = [
    ("R1", "Corta", "Alej", "Mant"), ("R2", "Corta", "Ig", "Fren"), ("R3", "Corta", "Acer", "Fren"),
    ("R4", "Media", "Alej", "Acel"), ("R5", "Media", "Ig", "Mant"), ("R6", "Media", "Acer", "Fren"),
    ("R7", "Larga", "Alej", "Acel"), ("R8", "Larga", "Ig", "Acel"), ("R9", "Larga", "Acer", "Mant"),
]

# ==============================================================================
# MOTORES DE INFERENCIA (SISO Y MISO)
# ==============================================================================
def siso_sugeno_mamdani(d_val: float):
    ac = float(fuzz.interp_membership(x_dist, d_corta, d_val))
    am = float(fuzz.interp_membership(x_dist, d_media, d_val))
    al = float(fuzz.interp_membership(x_dist, d_larga, d_val))

    z1 = np.clip(0.8 * d_val - 90, -100, 100)
    z2 = np.clip(0.4 * d_val - 20, -100, 100)
    z3 = np.clip(0.6 * d_val + 15, -100, 100)
    out_s = (ac * z1 + am * z2 + al * z3) / max(ac + am + al, 1e-9)

    agg = np.fmax(np.minimum(ac, c_fren), np.fmax(np.minimum(am, c_mant), np.minimum(al, c_acel)))
    out_m = float(fuzz.defuzz(z_ctrl, agg, 'centroid'))
    return out_s, out_m, (ac, am, al)

def evaluar_fam(d_val: float, v_val: float):
    md = {
        "Corta": float(fuzz.interp_membership(x_dist, d_corta, d_val)),
        "Media": float(fuzz.interp_membership(x_dist, d_media, d_val)),
        "Larga": float(fuzz.interp_membership(x_dist, d_larga, d_val)),
    }
    mv = {
        "Alej": float(fuzz.interp_membership(y_vrel, v_alej, v_val)),
        "Ig":   float(fuzz.interp_membership(y_vrel, v_ig,   v_val)),
        "Acer": float(fuzz.interp_membership(y_vrel, v_acer, v_val)),
    }
    return [(r, d, v, cons, min(md[d], mv[v])) for r, d, v, cons in FAM]

def miso_mamdani(d_val: float, v_val: float) -> float:
    agg = np.zeros_like(z_ctrl)
    mapa = {"Fren": c_fren, "Mant": c_mant, "Acel": c_acel}
    for _, _, _, cons, a in evaluar_fam(d_val, v_val):
        if a > 0:
            agg = np.fmax(agg, np.minimum(a, mapa[cons]))
    return float(fuzz.defuzz(z_ctrl, agg, 'centroid')) if np.sum(agg) > 0 else 0.0

def miso_sugeno(d_val: float, v_val: float) -> float:
    num, den = 0.0, 0.0
    for _, _, _, cons, a in evaluar_fam(d_val, v_val):
        if a > 0:
            if cons == "Fren":   zi = np.clip(0.5 * d_val - 1.2 * v_val - 60, -100, 100)
            elif cons == "Mant": zi = np.clip(0.2 * d_val - 0.5 * v_val,      -100, 100)
            else:                zi = np.clip(0.6 * d_val - 0.8 * v_val + 20, -100, 100)
            num += a * zi; den += a
    return num / max(den, 1e-9)

def miso_tsukamoto(d_val: float, v_val: float) -> float:
    num, den = 0.0, 0.0
    for _, _, _, cons, a in evaluar_fam(d_val, v_val):
        if a > 0:
            if cons == "Fren":   zi = -100.0 * a
            elif cons == "Mant": zi = -25.0 + 50.0 * a
            else:                zi = 100.0 * a
            num += a * zi; den += a
    return num / max(den, 1e-9)

@st.cache_data
def precalcular_curvas_y_superficies():
    d_1d = np.linspace(0, 100, 101)
    s_1d, m_1d = zip(*[siso_sugeno_mamdani(d)[:2] for d in d_1d])

    dg = np.linspace(0, 100, 22)
    vg = np.linspace(-30, 30, 22)
    D, V = np.meshgrid(dg, vg)
    ZM, ZS, ZT = np.zeros_like(D), np.zeros_like(D), np.zeros_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ZM[i, j] = miso_mamdani(D[i, j], V[i, j])
            ZS[i, j] = miso_sugeno(D[i, j], V[i, j])
            ZT[i, j] = miso_tsukamoto(D[i, j], V[i, j])
    return d_1d, np.array(s_1d), np.array(m_1d), D, V, ZM, ZS, ZT

d_1d, curva_s1d, curva_m1d, D_g, V_g, ZM, ZS, ZT = precalcular_curvas_y_superficies()

# ==============================================================================
# PESTAÑAS INTERACTIVAS
# ==============================================================================
tab1, tab2 = st.tabs([
    "1. SISO (1 Entrada): Sugeno vs. Mamdani",
    "2. MISO (2 Entradas): Sugeno vs. Mamdani vs. Tsukamoto"
])

with tab1:
    st.header("Modelo SISO: Distancia al Vehículo Frontal → Acción de Control")
    c1, c2 = st.columns([1, 2.2])
    with c1:
        d_in1 = st.slider("Distancia al vehículo frontal (m):", 0.0, 100.0, 32.0, 1.0, key="d1")
        out_s1, out_m1, (ac, am, al) = siso_sugeno_mamdani(d_in1)
        st.metric("Sugeno 1D (1.er Orden)", f"{out_s1:.2f} %")
        st.metric("Mamdani 1D (Centroide)", f"{out_m1:.2f} %")
        st.info(f"**Membresías Activas:**\n- Corta: {ac:.2f}\n- Media: {am:.2f}\n- Larga: {al:.2f}")

    with c2:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=d_1d, y=curva_s1d, name="Sugeno (1.er Orden)", line=dict(color="#ff7f0e", width=3)))
        fig1.add_trace(go.Scatter(x=d_1d, y=curva_m1d, name="Mamdani (Centroide)", line=dict(color="#1f77b4", width=3, dash="dash")))
        fig1.add_trace(go.Scatter(x=[d_in1, d_in1], y=[out_s1, out_m1], mode="markers", name="Punto Actual", marker=dict(size=10, color="red")))
        fig1.update_layout(
            title="Curvas de Transferencia SISO (Negativo = Frenar | Positivo = Acelerar)",
            xaxis_title="Distancia (m)", yaxis_title="Acción de Control (%)",
            template="plotly_white", height=420
        )
        st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.header("Modelo MISO: Distancia y Velocidad Relativa → Acción de Control")
    cs1, cs2 = st.columns(2)
    d_in2 = cs1.slider("Distancia al vehículo frontal (m):", 0.0, 100.0, 32.0, 1.0, key="d2")
    v_in2 = cs2.slider("Velocidad Relativa (km/h) [+ Acercándose / - Alejándose]:", -30.0, 30.0, 12.0, 1.0, key="v2")

    t0 = time.perf_counter(); out_s = miso_sugeno(d_in2, v_in2);    ts = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); out_m = miso_mamdani(d_in2, v_in2);   tm = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); out_t = miso_tsukamoto(d_in2, v_in2); tt = (time.perf_counter() - t0) * 1000

    k1, k2, k3 = st.columns(3)
    k1.metric("Sugeno (1.er Orden)", f"{out_s:.2f} %", f"{ts:.3f} ms", delta_color="off")
    k2.metric("Mamdani (Centroide)", f"{out_m:.2f} %", f"{tm:.3f} ms", delta_color="off")
    k3.metric("Tsukamoto (Monótono)", f"{out_t:.2f} %", f"{tt:.3f} ms", delta_color="off")

    g1, g2, g3 = st.columns(3)
    for col, Z_mat, titulo, pt in [
        (g1, ZS, "Superficie Sugeno (TSK)", out_s),
        (g2, ZM, "Superficie Mamdani", out_m),
        (g3, ZT, "Superficie Tsukamoto", out_t),
    ]:
        with col:
            fig3 = go.Figure(data=[
                go.Surface(z=Z_mat, x=D_g, y=V_g, colorscale="RdYlGn", showscale=False),
                go.Scatter3d(x=[d_in2], y=[v_in2], z=[pt], mode="markers", marker=dict(size=7, color="black"))
            ])
            fig3.update_layout(
                title=titulo, height=360, margin=dict(l=0, r=0, t=35, b=0),
                scene=dict(xaxis_title="Dist (m)", yaxis_title="VelRel", zaxis_title="Ctrl (%)")
            )
            st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Auditoría de Reglas Activas (Matriz FAM Exhaustiva)")
    fam_eval = evaluar_fam(d_in2, v_in2)
    df_fam = pd.DataFrame([
        {"Regla": r, "Antecedentes (AND)": f"{d} ∧ {v}", "Consecuente": c, "Fuerza de Disparo (α)": round(a, 3)}
        for r, d, v, c, a in fam_eval
    ])
    st.dataframe(df_fam, use_container_width=True, hide_index=True)
    
    