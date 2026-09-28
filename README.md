# MIA-ITEL: Artificial Intelligence Master

This repository contains coursework, exercises, and projects developed during the *Máster en Inteligencia Artificial (ITEL)* program.  
It includes implementations of supervised, unsupervised, and reinforcement learning algorithms, organized by units, along with datasets and supporting materials.

---

## 📂 Repository Structure

- **AprendizajeAutomaticoAvanzado_U1/**  
  Supervised (MLP classifier), Unsupervised (PCA + KMeans), Reinforcement (Q-learning)

- **AprendizajeAutomaticoAvanzado_U2/**  
  Advanced supervised learning and optimization techniques

- **AprendizajeAutomaticoAvanzado_U3/**  
  Deep learning workflows and neural networks

- **AprendizajeAutomaticoAvanzado_U4/**  
  Applied reinforcement learning and hybrid approaches

- **data/**  
  CSV datasets used for experiments

---

## ⚙️ Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
uv init
uv add pandas scikit-learn matplotlib seaborn nltk gymnasium
uv sync
