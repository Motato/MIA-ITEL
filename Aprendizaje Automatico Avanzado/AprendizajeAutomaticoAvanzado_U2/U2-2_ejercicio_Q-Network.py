# ==========================================
# Implementación de DQN para CartPole-v1
# ==========================================
import os
import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Fijar semillas para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Arquitectura de Red (Q-Network)
# ==========================================
class QNetwork(nn.Module):
    """
    MLP que toma un vector de estado continuo y devuelve el valor Q estimado
    para cada acción discreta posible: Q(s, a; theta).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

# Buffer de experiencia para descorrelacionar transiciones
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).unsqueeze(1).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 2. Agente DQN y Política Epsilon-Greedy
# ==========================================
class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.985, # Antes era 0.995
        buffer_capacity: int = 20_000,
        batch_size: int = 64,
        target_update_freq: int = 10
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Hiperparámetros de exploración
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Redes de política y objetivo
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss para estabilidad
        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def sync_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==========================================
# 3. Bucle de Entrenamiento
# ==========================================
def train_dqn(env_name: str = "CartPole-v1", episodes: int = 400):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    scores = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, float(done))
            agent.train_step()

            state = next_state
            total_reward += reward

        agent.update_epsilon()
        if episode % agent.target_update_freq == 0:
            agent.sync_target_network()

        scores.append(total_reward)
        avg_score = np.mean(scores[-50:])

        if episode % 20 == 0:
            print(f"Episodio {episode:3d} | Recompensa: {total_reward:5.1f} | Media (50 ep): {avg_score:5.1f} | Epsilon: {agent.epsilon:.3f}")

        # Continuamos entrenando hasta el final de los episodios para poder comparar justamente con Q-Learning
        # aunque el entorno se haya resuelto antes.

    env.close()
    return agent, scores

# ==========================================
# 4. Evaluación visual del agente entrenado
# ==========================================
def evaluate_dqn(agent, env_name="CartPole-v1", episodes=3):
    env = gym.make(env_name, render_mode="human")
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Política completamente greedy

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Evaluación visual - Episodio {episode}: Recompensa = {total_reward}")

    env.close()
    agent.epsilon = original_epsilon

# ==========================================
# Ejecución Principal
# ==========================================
if __name__ == "__main__":
    print("Iniciando entrenamiento de DQN...")
    trained_agent, history = train_dqn(episodes=400)
    
    # 1. Guardar los datos sin procesar para la gráfica comparativa de la Actividad 2.2
    np.save("dqn_rewards.npy", history)
    print("Recompensas guardadas en 'dqn_rewards.npy'")

    # 2. Evaluación visual (AQUÍ TOMA TUS CAPTURAS DE PANTALLA)
    print("\nIniciando evaluación visual. ¡Prepara tus capturas de pantalla!")
    evaluate_dqn(trained_agent)
    
    # 3. Graficar curva individual de DQN y guardar en resultados/
    os.makedirs('resultados', exist_ok=True)
    
    rolling_length = 50
    if len(history) >= rolling_length:
        avg_rewards = [np.mean(history[i:i+rolling_length]) for i in range(len(history)-rolling_length)]
        
        plt.figure(figsize=(10, 5))
        plt.plot(avg_rewards, label='Recompensa media (50 ep)', color='blue')
        plt.title('Gráfica Q-Learning_CartPole')
        plt.xlabel('Episodios')
        plt.ylabel('Recompensa Media')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.savefig('resultados/dqn_Captura_Visual_QLearning.png')
        print("\nGráfica guardada en 'resultados/dqn_Captura_Visual_QLearning.png'")
        plt.show()
        
        