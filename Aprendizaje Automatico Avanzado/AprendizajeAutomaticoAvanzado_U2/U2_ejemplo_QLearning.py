# ============================================================
# Implementación en Python de Q-Learning en FrozenLake
# ============================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1) Configuración del entorno y Q-table
# ------------------------------------------------------------
# is_slippery=False → entorno determinista (movimientos no fallan).
# observation_space.n → número de estados posibles.
# action_space.n → número de acciones posibles.
# ============================================================
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# ============================================================
# 2) Hiperparámetros
# ------------------------------------------------------------
learning_rate = 0.1     # tasa de aprendizaje (α)
gamma         = 0.99    # factor de descuento (γ)
epsilon       = 1.0     # probabilidad inicial de exploración
epsilon_decay = 0.001   # decremento de epsilon por episodio
episodes      = 2000    # número de episodios de entrenamiento

# Lista para almacenar recompensas por episodio
rewards_per_episode = []

# ============================================================
# 3) Entrenamiento Q-Learning
# ------------------------------------------------------------
# Regla de actualización:
# Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
# ============================================================
for i in range(episodes):
    state, _ = env.reset()
    done = False
    total_rewards = 0

    while not done:
        # Selección de acción (Epsilon-greedy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # explorar
        else:
            action = np.argmax(q_table[state, :])  # explotar

        # Ejecutar acción en el entorno
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Actualización de Q-Value
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        # Avanzar al siguiente estado
        state = new_state
        total_rewards += reward

    # Decaimiento de exploración
    epsilon = max(0.01, epsilon - epsilon_decay)

    # Guardar recompensa total del episodio
    rewards_per_episode.append(total_rewards)

# ============================================================
# 4) Cálculo del promedio móvil (cada 100 episodios)
# ------------------------------------------------------------
# Esto suaviza la curva para ver la tendencia de aprendizaje.
# ============================================================
rolling_length = 100
average_reward = [
    np.mean(rewards_per_episode[i:i+rolling_length])
    for i in range(len(rewards_per_episode) - rolling_length)
]

# ============================================================
# 5) Graficación de la curva de aprendizaje
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(average_reward, color='blue', label='Promedio móvil (100 eps)')
plt.title('Curva de Aprendizaje - FrozenLake (Q-Learning)')
plt.xlabel('Episodios')
plt.ylabel('Recompensa Media')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

