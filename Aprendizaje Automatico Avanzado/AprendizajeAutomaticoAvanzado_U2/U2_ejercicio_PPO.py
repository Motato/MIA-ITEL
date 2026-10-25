import gymnasium as gym
from stable_baselines3 import PPO

# 1. Crear el entorno (ej. LunarLander o BipedalWalker para control continuo)
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Instanciar el agente PPO (MlpPolicy es la arquitectura Actor-Crítico para vectores)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64)

# 3. Entrenar al agente
print("Entrenando PPO...")
model.learn(total_timesteps=15000)

# 4. Guardar el modelo en la carpeta de resultados
model.save("resultados/ppo_lunarlander")
env.close()