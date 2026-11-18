import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Definición de la Función de Membresía Base ---
def triangular(x, a, b, c):
    """Calcula la función de membresía triangular."""
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

# --- Configuración de la Interfaz ---
st.set_page_config(page_title="Operaciones Difusas Avanzadas", layout="wide")
st.title("Análisis Algorítmico de Operaciones de Conjuntos Difusos")
st.markdown("Visualización de las operaciones matemáticas **Unión (OR)** e **Intersección (AND)** aplicadas a dos conceptos imprecisos.")

# Universo de discurso
x = np.linspace(0, 100, 500)

# --- Controles de Usuario ---
st.sidebar.header("Calibración de Conjuntos")
st.sidebar.subheader("Conjunto A (Azul)")
a_start = st.sidebar.slider("A: Inicio", 0, 100, 10, key="a1")
a_peak = st.sidebar.slider("A: Pico", 0, 100, 30, key="a2")
a_end = st.sidebar.slider("A: Fin", 0, 100, 60, key="a3")

st.sidebar.subheader("Conjunto B (Rojo)")
b_start = st.sidebar.slider("B: Inicio", 0, 100, 40, key="b1")
b_peak = st.sidebar.slider("B: Pico", 0, 100, 70, key="b2")
b_end = st.sidebar.slider("B: Fin", 0, 100, 90, key="b3")

# --- Cálculo de Algoritmos Difusos ---
mu_A = triangular(x, a_start, a_peak, a_end)
mu_B = triangular(x, b_start, b_peak, b_end)

# Operaciones: Intersección (Min) y Unión (Max)
mu_interseccion = np.minimum(mu_A, mu_B)
mu_union = np.maximum(mu_A, mu_B)

# --- Renderizado Gráfico ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.tight_layout(pad=4.0)

# Gráfica 1: Conjuntos Originales
axs[0].plot(x, mu_A, color='blue', label='Conjunto A', linewidth=2)
axs[0].plot(x, mu_B, color='red', label='Conjunto B', linewidth=2)
axs[0].set_title("Conjuntos Difusos A y B")
axs[0].legend()

# Gráfica 2: Intersección (AND)
axs[1].plot(x, mu_A, color='blue', linestyle='--', alpha=0.3)
axs[1].plot(x, mu_B, color='red', linestyle='--', alpha=0.3)
axs[1].plot(x, mu_interseccion, color='purple', label='A AND B (Min)', linewidth=3)
axs[1].fill_between(x, 0, mu_interseccion, color='purple', alpha=0.3)
axs[1].set_title("Operación: Intersección (Mínimo)")
axs[1].legend()

# Gráfica 3: Unión (OR)
axs[2].plot(x, mu_A, color='blue', linestyle='--', alpha=0.3)
axs[2].plot(x, mu_B, color='red', linestyle='--', alpha=0.3)
axs[2].plot(x, mu_union, color='green', label='A OR B (Max)', linewidth=3)
axs[2].fill_between(x, 0, mu_union, color='green', alpha=0.2)
axs[2].set_title("Operación: Unión (Máximo)")
axs[2].legend()

for ax in axs:
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Grado de Membresía (\u03bc)")
    ax.set_xlabel("Universo de Discurso")
    ax.grid(True, linestyle='--', alpha=0.6)

st.pyplot(fig)
