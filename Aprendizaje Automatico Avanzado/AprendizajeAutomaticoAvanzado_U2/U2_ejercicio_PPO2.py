import gymnasium as gym
from stable_baselines3 import PPO
import os

# Asegurar que exista la carpeta de resultados
os.makedirs('resultados', exist_ok=True)

# 1. Crear el entorno continuo (BipedalWalker-v3 o LunarLander-v2)
# Nota: Si usas BipedalWalker, quizás necesites instalar box2d: uv pip install "gymnasium[box2d]"
env_name = "BipedalWalker-v3"
env = gym.make(env_name)

# 2. Instanciar el agente PPO (Actor-Crítico con política MLP)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=3e-4, 
    n_steps=2048, 
    batch_size=64,
    tensorboard_log="./resultados/ppo_tensorboard/" # Guarda métricas
)

# 3. Entrenar al agente
print("Entrenando PPO...")
model.learn(total_timesteps=50000)

# 4. Guardar el modelo entrenado
model.save("resultados/ppo_bipedalwalker")
env.close()

# 5. Evaluación visual para capturas de pantalla
print("Iniciando evaluación visual...")
env_eval = gym.make(env_name, render_mode="human")
obs, info = env_eval.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_eval.step(action)
    
    if terminated or truncated:
        obs, info = env_eval.reset()

env_eval.close()

