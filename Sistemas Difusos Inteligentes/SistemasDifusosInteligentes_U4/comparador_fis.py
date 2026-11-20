import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import skfuzzy as fuzz
from skfuzzy import control as ctrl

st.set_page_config(page_title="Comparador FIS: Mamdani, Sugeno y Tsukamoto", layout="wide")
st.title("Laboratorio Comparativo de Modelos de Inferencia Difusa (HVAC)")
st.markdown("Simulación interactiva de **Mamdani (1 y 2 entradas)**, **Sugeno (1.er orden)** y **Tsukamoto (Inversión Monótona)**.")

# ==============================================================================
# 1. DEFINICIÓN DE UNIVERSOS Y FUNCIONES DE MEMBRESÍA COMPARTIDAS (2 ENTRADAS)
# ==============================================================================
x_temp = np.linspace(15, 35, 201)
y_hum  = np.linspace(20, 90, 201)
z_pot  = np.linspace(0, 100, 201)

# Membresías de Temperatura [15, 35] °C
t_fria = fuzz.trapmf(x_temp, [15, 15, 19, 23])
t_conf = fuzz.trimf(x_temp,  [20, 24, 28])
t_cal  = fuzz.trapmf(x_temp, [25, 29, 35, 35])

# Membresías de Humedad [20, 90] %
h_seca = fuzz.trapmf(y_hum, [20, 20, 35, 50])
h_norm = fuzz.trimf(y_hum,  [35, 55, 75])
h_hum  = fuzz.trapmf(y_hum, [60, 75, 90, 90])

# Membresías de Salida Mamdani [0, 100] %
p_baja  = fuzz.trimf(z_pot, [0, 0, 45])
p_media = fuzz.trimf(z_pot, [25, 50, 75])
p_alta  = fuzz.trimf(z_pot, [55, 100, 100])

# Matriz FAM (9 reglas): (Temp, Hum, Consecuente)
FAM_RULES = [
    ("Fria", "Seca", "Baja"),   ("Fria", "Norm", "Baja"),   ("Fria", "Hum", "Media"),
    ("Conf", "Seca", "Baja"),   ("Conf", "Norm", "Media"),  ("Conf", "Hum", "Alta"),
    ("Cal",  "Seca", "Media"),  ("Cal",  "Norm", "Alta"),   ("Cal",  "Hum", "Alta"),
]

def evaluar_antecedentes(t_val: float, h_val: float):
    """Calcula grados de membresía y activación (alpha) de las 9 reglas."""
    mt = {
        "Fria": float(fuzz.interp_membership(x_temp, t_fria, t_val)),
        "Conf": float(fuzz.interp_membership(x_temp, t_conf, t_val)),
        "Cal":  float(fuzz.interp_membership(x_temp, t_cal,  t_val)),
    }
    mh = {
        "Seca": float(fuzz.interp_membership(y_hum, h_seca, h_val)),
        "Norm": float(fuzz.interp_membership(y_hum, h_norm, h_val)),
        "Hum":  float(fuzz.interp_membership(y_hum, h_hum,  h_val)),
    }
    alphas = [(min(mt[t], mh[h]), cons) for t, h, cons in FAM_RULES]
    return alphas

# ==============================================================================
# 2. MOTORES DE INFERENCIA: MAMDANI, SUGENO Y TSUKAMOTO
# ==============================================================================
def inferir_mamdani(t_val: float, h_val: float) -> float:
    alphas = evaluar_antecedentes(t_val, h_val)
    agregado = np.zeros_like(z_pot)
    mapa_mf = {"Baja": p_baja, "Media": p_media, "Alta": p_alta}
    for alpha, cons in alphas:
        if alpha > 0:
            agregado = np.fmax(agregado, np.minimum(alpha, mapa_mf[cons]))
    return float(fuzz.defuzz(z_pot, agregado, 'centroid')) if np.sum(agregado) > 0 else 0.0

def inferir_sugeno(t_val: float, h_val: float) -> float:
    """Sugeno de 1.er orden: z_i = p_i*T + q_i*H + r_i."""
    alphas = evaluar_antecedentes(t_val, h_val)
    num, den = 0.0, 0.0
    for alpha, cons in alphas:
        if alpha > 0:
            if cons == "Baja":
                z_i = np.clip(0.5 * t_val + 0.1 * h_val, 0, 100)
            elif cons == "Media":
                z_i = np.clip(1.2 * t_val + 0.3 * h_val + 5.0, 0, 100)
            else:
                z_i = np.clip(2.0 * t_val + 0.4 * h_val + 15.0, 0, 100)
            num += alpha * z_i
            den += alpha
    return num / den if den > 0 else 0.0

def inferir_tsukamoto(t_val: float, h_val: float) -> float:
    """Tsukamoto con inversión exacta de funciones monótonas lineales."""
    alphas = evaluar_antecedentes(t_val, h_val)
    num, den = 0.0, 0.0
    for alpha, cons in alphas:
        if alpha > 0:
            if cons == "Baja":
                z_i = 50.0 * (1.0 - alpha)   # Monótona decreciente [0, 50]
            elif cons == "Media":
                z_i = 25.0 + 50.0 * alpha    # Monótona creciente [25, 75]
            else:
                z_i = 50.0 + 50.0 * alpha    # Monótona creciente [50, 100]
            num += alpha * z_i
            den += alpha
    return num / den if den > 0 else 0.0

@st.cache_data
def generar_superficies_3d():
    t_grid = np.linspace(15, 35, 22)
    h_grid = np.linspace(20, 90, 22)
    T, H = np.meshgrid(t_grid, h_grid)
    ZM, ZS, ZT = np.zeros_like(T), np.zeros_like(T), np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            ZM[i, j] = inferir_mamdani(T[i, j], H[i, j])
            ZS[i, j] = inferir_sugeno(T[i, j], H[i, j])
            ZT[i, j] = inferir_tsukamoto(T[i, j], H[i, j])
    return T, H, ZM, ZS, ZT

# ==============================================================================
# 3. INTERFAZ POR PESTAÑAS
# ==============================================================================
tab1, tab2 = st.tabs([
    "1. Mamdani SISO (1 Entrada → 1 Salida)",
    "2. Comparativa MISO: Mamdani vs. Sugeno vs. Tsukamoto"
])

# --- PESTAÑA 1: MAMDANI 1 ENTRADA ---
with tab1:
    st.header("Modelo Mamdani: 1 Variable de Entrada (Error Térmico)")
    err_u = np.linspace(-10, 10, 201)
    e_neg = fuzz.trapmf(err_u, [-10, -10, -4, 0])
    e_cer = fuzz.trimf(err_u,  [-3, 0, 3])
    e_pos = fuzz.trapmf(err_u, [0, 4, 10, 10])

    c1, c2 = st.columns([1, 2])
    with c1:
        err_in = st.slider("Error de Temperatura (T_actual - T_deseada) °C:", -10.0, 10.0, 3.5, 0.5)
        a_neg = float(fuzz.interp_membership(err_u, e_neg, err_in))
        a_cer = float(fuzz.interp_membership(err_u, e_cer, err_in))
        a_pos = float(fuzz.interp_membership(err_u, e_pos, err_in))

        agg_1d = np.fmax(np.minimum(a_neg, p_baja), np.fmax(np.minimum(a_cer, p_media), np.minimum(a_pos, p_alta)))
        pot_1d = float(fuzz.defuzz(z_pot, agg_1d, 'centroid'))
        st.metric("Potencia del Compresor (Mamdani 1D)", f"{pot_1d:.2f} %")
        st.write(f"**Activaciones:** Negativo={a_neg:.2f} | Cero={a_cer:.2f} | Positivo={a_pos:.2f}")

    with c2:
        # Curva de transferencia 1D completa
        curva_out = []
        for e in err_u[::4]:
            an = fuzz.interp_membership(err_u, e_neg, e)
            ac = fuzz.interp_membership(err_u, e_cer, e)
            ap = fuzz.interp_membership(err_u, e_pos, e)
            ag = np.fmax(np.minimum(an, p_baja), np.fmax(np.minimum(ac, p_media), np.minimum(ap, p_alta)))
            curva_out.append(fuzz.defuzz(z_pot, ag, 'centroid'))

        fig1d = go.Figure()
        fig1d.add_trace(go.Scatter(x=err_u[::4], y=curva_out, mode='lines', name='Curva Mamdani', line=dict(color='teal', width=3)))
        fig1d.add_trace(go.Scatter(x=[err_in], y=[pot_1d], mode='markers', name='Punto Actual', marker=dict(size=12, color='red')))
        fig1d.update_layout(
            title="Curva de Transferencia No Lineal (1 Entrada → 1 Salida)",
            xaxis_title="Error de Temperatura (°C)", yaxis_title="Potencia (%)",
            template="plotly_white", height=380
        )
        st.plotly_chart(fig1d, use_container_width=True)

# --- PESTAÑA 2: COMPARATIVA MISO (3 MODELOS) ---
with tab2:
    st.header("Comparación Directa: Mamdani vs. Sugeno vs. Tsukamoto (2 Entradas)")
    col_s1, col_s2 = st.columns(2)
    t_val = col_s1.slider("Temperatura Interior (°C):", 15.0, 35.0, 27.5, 0.5)
    h_val = col_s2.slider("Humedad Relativa (%):", 20.0, 90.0, 68.0, 1.0)

    # Medición de tiempos de cómputo (Benchmark en ms)
    t0 = time.perf_counter(); out_m = inferir_mamdani(t_val, h_val);   tm = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); out_s = inferir_sugeno(t_val, h_val);    ts = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); out_t = inferir_tsukamoto(t_val, h_val); tt = (time.perf_counter() - t0) * 1000

    m1, m2, m3 = st.columns(3)
    m1.metric("Mamdani (Centroide)", f"{out_m:.2f} %", f"{tm:.3f} ms", delta_color="off")
    m2.metric("Sugeno (1.er Orden)", f"{out_s:.2f} %", f"{ts:.3f} ms", delta_color="off")
    m3.metric("Tsukamoto (Monótono)", f"{out_t:.2f} %", f"{tt:.3f} ms", delta_color="off")

    # Superficies 3D lado a lado
    T_g, H_g, ZM, ZS, ZT = generar_superficies_3d()
    g1, g2, g3 = st.columns(3)
    for col, Z_mat, titulo, out_pt in [
        (g1, ZM, "Superficie Mamdani", out_m),
        (g2, ZS, "Superficie Sugeno (TSK)", out_s),
        (g3, ZT, "Superficie Tsukamoto", out_t),
    ]:
        with col:
            fig = go.Figure(data=[
                go.Surface(z=Z_mat, x=T_g, y=H_g, colorscale="Viridis", showscale=False),
                go.Scatter3d(x=[t_val], y=[h_val], z=[out_pt], mode="markers", marker=dict(size=6, color="red"))
            ])
            fig.update_layout(
                title=titulo, height=380, margin=dict(l=0, r=0, t=35, b=0),
                scene=dict(xaxis_title="Temp (°C)", yaxis_title="Hum (%)", zaxis_title="Pot (%)")
            )
            st.plotly_chart(fig, use_container_width=True)