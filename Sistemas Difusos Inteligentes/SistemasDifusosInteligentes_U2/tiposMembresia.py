import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Mathematical Definitions ---
def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def trapezoidal(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-9), 1), (d - x) / (d - c + 1e-9)))

def gaussian(x, center, sigma):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)

def sigmoidal(x, a, center):
    return 1 / (1 + np.exp(-a * (x - center)))

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Fuzzy Logic Visualizer", layout="wide")
st.title("Interactive Fuzzy Logic Membership Functions")
st.write("Adjust the parameters in the sidebar to see how the membership functions change in real-time.")

# Universe of discourse (x-axis)
x = np.linspace(0, 100, 500)

# Sidebar Inputs
st.sidebar.header("Function Parameters")

st.sidebar.subheader("Triangular")
tri_a = st.sidebar.slider("Start (a)", 0, 100, 20, key="tri_a")
tri_b = st.sidebar.slider("Peak (b)", 0, 100, 50, key="tri_b")
tri_c = st.sidebar.slider("End (c)", 0, 100, 80, key="tri_c")

st.sidebar.subheader("Trapezoidal")
trap_a = st.sidebar.slider("Start (a)", 0, 100, 10, key="trap_a")
trap_b = st.sidebar.slider("Flat Start (b)", 0, 100, 30, key="trap_b")
trap_c = st.sidebar.slider("Flat End (c)", 0, 100, 60, key="trap_c")
trap_d = st.sidebar.slider("End (d)", 0, 100, 90, key="trap_d")

st.sidebar.subheader("Gaussian")
gauss_c = st.sidebar.slider("Center", 0, 100, 50, key="gauss_c")
gauss_sig = st.sidebar.slider("Width (Sigma)", 1, 30, 15, key="gauss_sig")

st.sidebar.subheader("Sigmoidal")
sig_c = st.sidebar.slider("Center (Cross point)", 0, 100, 50, key="sig_c")
sig_a = st.sidebar.slider("Slope (a)", -1.0, 1.0, 0.2, key="sig_a")

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.tight_layout(pad=4.0)

# Triangular Plot
axs[0, 0].plot(x, triangular(x, tri_a, tri_b, tri_c), color='blue')
axs[0, 0].set_title("Triangular")
axs[0, 0].set_ylim(-0.1, 1.1)

# Trapezoidal Plot
axs[0, 1].plot(x, trapezoidal(x, trap_a, trap_b, trap_c, trap_d), color='green')
axs[0, 1].set_title("Trapezoidal")
axs[0, 1].set_ylim(-0.1, 1.1)

# Gaussian Plot
axs[1, 0].plot(x, gaussian(x, gauss_c, gauss_sig), color='red')
axs[1, 0].set_title("Gaussian")
axs[1, 0].set_ylim(-0.1, 1.1)

# Sigmoidal Plot
axs[1, 1].plot(x, sigmoidal(x, sig_a, sig_c), color='purple')
axs[1, 1].set_title("Sigmoidal")
axs[1, 1].set_ylim(-0.1, 1.1)

for ax in axs.flat:
    ax.set_ylabel("Degree of Membership")
    ax.set_xlabel("Universe of Discourse")
    ax.grid(True, linestyle='--', alpha=0.6)

st.pyplot(fig)
