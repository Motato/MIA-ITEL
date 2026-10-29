# 📘 Aprendizaje Automático Avanzado

Este repositorio contiene ejemplos y ejercicios de aprendizaje automático avanzado, organizados por unidades (U1–U4).  
Incluye prácticas de aprendizaje supervisado, no supervisado, y por refuerzo, con librerías modernas de Python.

---

## 📂 Estructura del proyecto



---

## ⚙️ Configuración del entorno

Este proyecto usa [uv](https://github.com/astral-sh/uv) para la gestión de dependencias y entornos virtuales.

1. **Instalar uv** (si no lo tienes):
   ```bash
   pip install uv


" Unidad 1:
PYTHONPATH=.
PROJECT_NAME=aprendizaje-automatico-avanzado

DATASET_BREAST_CANCER=Aprendizaje Automatico Avanzado/data/breast_cancer_dataset.csv
DATASET_CIBERSUICIDIO=Aprendizaje Automatico Avanzado/data/dataset_cibersuicidio.csv
DATASET_REVIEWS=Aprendizaje Automatico Avanzado/data/reviews_sentiment.csv


uv sync --active



uv run aaa


Proyecto Aprendizaje Automático Avanzado funcionando correctamente


uv run python "Aprendizaje Automatico Avanzado/AprendizajeAutomaticoAvanzado_U1/U1_ejemplo.py"




# Unidad 2: 
# Reinforcement Learning with DQN and PPO

## Description

This project contains two Reinforcement Learning implementations developed for the **Advanced Machine Learning** course.

The objective is to compare two different approaches for solving control problems in Gymnasium environments:

- **Deep Q-Network (DQN)** implemented from scratch using PyTorch.
- **Proximal Policy Optimization (PPO)** using Stable-Baselines3.

---

## Project Structure

```text
MIA-ITEL/
│
├── Aprendizaje Automatico Avanzado/
│   │
│   ├── U2/
│   │   ├── U2_ejercicio_DQN.py
│   │   ├── U2_ejercicio_PPO.py
│   │   └── resultados/
│   │
│   ├── U1/
│   ├── U3/
│   └── U4/
│
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

---

# Script 1: Deep Q-Network (DQN)

**File:**

```text
U2_ejercicio_DQN.py
```

## Description

This script implements a Deep Q-Network (DQN) agent using PyTorch.

The implementation includes:

- Q-Network (Neural Network)
- Experience Replay Buffer
- Target Network
- Epsilon-Greedy Exploration
- Bellman Equation Optimization
- Learning Curve Visualization

## Environment

```text
CartPole-v1
```

## Output

- Trained DQN Agent
- Learning Curve
- Evaluation Results
- CartPole Visualization

---

# Script 2: Proximal Policy Optimization (PPO)

**File:**

```text
U2_ejercicio_PPO.py
```

## Description

This script trains an agent using the PPO (Proximal Policy Optimization) algorithm provided by Stable-Baselines3.

The implementation includes:

- Actor-Critic Architecture
- MLP Policy Network
- Continuous Control Environment
- Model Persistence
- Visual Evaluation

## Environment

```text
BipedalWalker-v3
```

## Output

- Trained PPO Model
- Saved Agent
- Visual Evaluation
- Screenshots for Academic Report

---

# Environment Setup

## Install uv

Official documentation:

https://docs.astral.sh/uv/

Verify installation:

```bash
uv --version
```

---

# Python Version

This project uses Python 3.13.

Install Python:

```bash
uv python install 3.13
```

Pin the project version:

```bash
uv python pin 3.13
```

Verify:

```bash
uv run python --version
```

Expected output:

```text
Python 3.13.x
```

---

# Install Dependencies

## Create Environment

```bash
uv sync
```

---

## Scientific Computing

```bash
uv add numpy
uv add matplotlib
```

---

## PyTorch

```bash
uv add torch
uv add torchvision
uv add torchaudio
```

Verify:

```bash
uv run python -c "import torch; print(torch.__version__)"
```

---

## Gymnasium

```bash
uv add gymnasium
```

Verify:

```bash
uv run python -c "import gymnasium; print('Gymnasium OK')"
```

---

## Box2D Environments

Required for:

- LunarLander
- BipedalWalker

Install:

```bash
uv add swig
uv add "gymnasium[box2d]"
```

Verify:

```bash
uv run python -c "import Box2D; print('Box2D OK')"
```

---

## Stable-Baselines3

Install:

```bash
uv add stable-baselines3
```

Verify:

```bash
uv run python -c "from stable_baselines3 import PPO; print('PPO OK')"
```

---

# Execute DQN

```bash
uv run python U2_ejercicio_DQN.py
```

---

# Execute PPO

```bash
uv run python U2_ejercicio_PPO.py
```

---

# Results Folder

The generated results are stored in:

```text
resultados/
```

Examples:

```text
resultados/
├── dqn_learning_curve.png
├── ppo_bipedalwalker.zip
└── capturas/
```

---

# Reinforcement Learning Concepts

## DQN

- Deep Q-Network
- Bellman Equation
- Replay Buffer
- Target Network
- Epsilon-Greedy Strategy

## PPO

- Policy Gradient
- Actor-Critic
- Advantage Estimation
- Clipped Objective Function
- Continuous Control

---

# Author

**Hernán Iván Montoya Motato**

Master's Degree in Artificial Intelligence

Advanced Machine Learning Course / Aprendizaje Automático Avanzado

ITEL