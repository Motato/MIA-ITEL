import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math

# 1. Configuración del entorno
env = gym.make('CartPole-v1')

# 2. Discretización de los estados (Buckets)
buckets = (1, 1, 6, 12)  # Ignoramos posición y velocidad del carro, nos enfocamos en el ángulo del poste
q_table = np.zeros(buckets + (env.action_space.n,))

def discretize_state(state):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(state))]
    return tuple(new_obs)

# 3. Hiperparámetros
episodes = 400 # Mismo número que usarás en DQN
learning_rate = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

q_rewards = []

# 4. Entrenamiento Q-Learning
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_discrete = discretize_state(next_state)

        # Actualización de Bellman
        q_table[state][action] += learning_rate * (reward + gamma * np.max(q_table[next_state_discrete]) - q_table[state][action])
        
        state = next_state_discrete
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    q_rewards.append(total_reward)

env.close()

# Guardar los resultados para compararlos luego
np.save("q_learning_rewards.npy", q_rewards)
print("Entrenamiento Q-Learning terminado.")