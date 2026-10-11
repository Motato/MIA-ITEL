import numpy as np
import gymnasium as gym

# ============================================================
# 1) Crear entorno FrozenLake
# ------------------------------------------------------------
# is_slippery=True → entorno estocástico (más realista, movimientos pueden fallar).
# observation_space.n → número de estados posibles (casillas del lago).
# action_space.n → número de acciones posibles (arriba, abajo, izquierda, derecha).
# ============================================================
env = gym.make("FrozenLake-v1", is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# ============================================================
# 2) Inicializar Q-table
# ------------------------------------------------------------
# Matriz Q de tamaño [n_states x n_actions], inicializada en ceros.
# Cada celda representa el valor esperado de tomar una acción en un estado.
# ============================================================
Q = np.zeros((n_states, n_actions))

# ============================================================
# 3) Hiperparámetros
# ------------------------------------------------------------
alpha        = 0.10   # tasa de aprendizaje
gamma        = 0.99   # factor de descuento (importancia de recompensas futuras)
epsilon      = 1.0    # probabilidad inicial de exploración
epsilon_min  = 0.05   # mínimo valor de epsilon
epsilon_decay= 0.999  # factor de decaimiento de epsilon por episodio
episodes     = 15000  # número de episodios de entrenamiento
max_steps    = 200    # pasos máximos por episodio
rng          = np.random.default_rng(42)  # generador aleatorio reproducible

# ============================================================
# 4) Política epsilon-greedy
# ------------------------------------------------------------
# Con probabilidad epsilon → explorar (acción aleatoria).
# Con probabilidad (1-epsilon) → explotar (acción con mayor valor Q).
# ============================================================
def epsilon_greedy(state, eps):
    if rng.random() < eps:
        return env.action_space.sample()
    return int(np.argmax(Q[state]))

# ============================================================
# 5) Entrenamiento Q-Learning
# ------------------------------------------------------------
# Actualización de la Q-table usando la regla de Bellman:
# Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
# ============================================================
for ep in range(episodes):
    state, _ = env.reset()
    done = False

    for _ in range(max_steps):
        # Selección de acción
        action = epsilon_greedy(state, epsilon)

        # Ejecutar acción en el entorno
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Actualización Q-Learning
        best_next = np.max(Q[next_state])
        td_target = reward + gamma * best_next
        td_error  = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state
        if done:
            break

    # Decaimiento de exploración
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# ============================================================
# 6) Evaluación del agente (sin exploración)
# ------------------------------------------------------------
# Ejecutamos varios episodios usando la política aprendida (argmax Q).
# Medimos la tasa de éxito (llegar a la meta).
# ============================================================
test_episodes = 200
success = 0

for _ in range(test_episodes):
    state, _ = env.reset()
    for _ in range(max_steps):
        action = int(np.argmax(Q[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            success += int(reward == 1.0)
            break

print(f"Tasa de éxito aproximada: {success}/{test_episodes} = {success/test_episodes:.2%}")

env.close()
