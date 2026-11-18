import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Definiciones Matemáticas Vectorizadas ---
def triangular(x, a, b, c):
    """Calcula la función de membresía triangular."""
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def trapezoidal(x, a, b, c, d):
    """Calcula la función de membresía trapezoidal."""
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-9), 1), (d - x) / (d - c + 1e-9)))

def gaussian(x, center, sigma):
    """Calcula la función de membresía gaussiana."""
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)

def sigmoidal(x, a, center):
    """Calcula la función de membresía sigmoidal."""
    return 1 / (1 + np.exp(-a * (x - center)))

# --- Funciones Auxiliares para Graficar con Plotly ---
def create_plot(x, y, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=title, line=dict(color=color, width=3), fill='tozeroy'))
    fig.update_layout(
        title=f"Conjunto Difuso: {title}",
        xaxis_title="Universo de Discurso (x)",
        yaxis_title="Grado de Membresía (\u03bc)",
        yaxis=dict(range=[-0.05, 1.1]),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# --- Configuración de la Interfaz Streamlit ---
st.set_page_config(page_title="Simulador de Lógica Difusa", layout="wide")
st.title("Visualizador Interactivo de Funciones de Membresía")
st.markdown("Herramienta interactiva para el análisis matemático de **Conjuntos Difusos**. Ajuste los parámetros en el panel lateral para visualizar el comportamiento de las fronteras imprecisas.")

# Universo de discurso (Eje X) global
x = np.linspace(0, 100, 500)

# --- Panel Lateral: Calibración de Parámetros ---
st.sidebar.header("Parámetros de las Funciones")

st.sidebar.subheader("1. Triangular")
tri_a = st.sidebar.slider("Inicio (a)", 0, 100, 20, key="tri_a")
tri_b = st.sidebar.slider("Pico Central (b)", 0, 100, 50, key="tri_b")
tri_c = st.sidebar.slider("Fin (c)", 0, 100, 80, key="tri_c")

st.sidebar.subheader("2. Trapezoidal")
trap_a = st.sidebar.slider("Inicio (a)", 0, 100, 10, key="trap_a")
trap_b = st.sidebar.slider("Inicio Meseta (b)", 0, 100, 30, key="trap_b")
trap_c = st.sidebar.slider("Fin Meseta (c)", 0, 100, 60, key="trap_c")
trap_d = st.sidebar.slider("Fin (d)", 0, 100, 90, key="trap_d")

st.sidebar.subheader("3. Gaussiana")
gauss_c = st.sidebar.slider("Centro (\u03bc)", 0, 100, 50, key="gauss_c")
gauss_sig = st.sidebar.slider("Anchura (\u03c3)", 1, 30, 15, key="gauss_sig")

st.sidebar.subheader("4. Sigmoidal")
sig_c = st.sidebar.slider("Punto de Cruce (c)", 0, 100, 50, key="sig_c")
sig_a = st.sidebar.slider("Pendiente (a)", -1.0, 1.0, 0.2, key="sig_a")

# --- Contenido Principal: Pestañas (Tabs) ---
tab1, tab2, tab3, tab4 = st.tabs(["Triangular", "Trapezoidal", "Gaussiana", "Sigmoidal"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Función Triangular")
        st.write("Alcanza su grado máximo de membresía (1) en un único punto central, formando un pico agudo. Ideal para evaluaciones estrictas de un solo valor ideal.")
        st.latex(r"\mu(x) = \max\left(0, \min\left(\frac{x-a}{b-a}, \frac{c-x}{c-b}\right)\right)")
    with col2:
        st.plotly_chart(create_plot(x, triangular(x, tri_a, tri_b, tri_c), "Triangular", "#1f77b4"), use_container_width=True)

with tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Función Trapezoidal")
        st.write("Presenta una cima plana que representa un rango continuo y absoluto de pertenencia total. Útil cuando un concepto es válido en un intervalo completo.")
        st.latex(r"\mu(x) = \max\left(0, \min\left(\frac{x-a}{b-a}, 1, \frac{d-x}{d-c}\right)\right)")
    with col2:
        st.plotly_chart(create_plot(x, trapezoidal(x, trap_a, trap_b, trap_c, trap_d), "Trapezoidal", "#2ca02c"), use_container_width=True)

with tab3:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Función Gaussiana")
        st.write("Curva simétrica en forma de campana suave. Ideal para modelar transiciones continuas y fenómenos naturales sin cortes abruptos.")
        st.latex(r"\mu(x) = e^{-\frac{1}{2}\left(\frac{x-c}{\sigma}\right)^2}")
    with col2:
        st.plotly_chart(create_plot(x, gaussian(x, gauss_c, gauss_sig), "Gaussiana", "#d62728"), use_container_width=True)

with tab4:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Función Sigmoidal")
        st.write("Curva asimétrica en forma de 'S'. Utilizada para umbrales abiertos o extremos (ej. saturación de sensores o límites de control).")
        st.latex(r"\mu(x) = \frac{1}{1 + e^{-a(x-c)}}")
    with col2:
        st.plotly_chart(create_plot(x, sigmoidal(x, sig_a, sig_c), "Sigmoidal", "#9467bd"), use_container_width=True)
        
        