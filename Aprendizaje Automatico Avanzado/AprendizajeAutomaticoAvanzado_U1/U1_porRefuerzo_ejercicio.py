# ============================================================
# reinforcement/custom_nlp_rl.py
# Q-Learning Tabular en un Entorno NLP Personalizado
# ============================================================

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ============================================================
# 1) Definición del Entorno Personalizado (Gymnasium)
# ============================================================
class SentimentEnv(gym.Env):
    """
    Convierte un dataset estático de NLP en un MDP para Q-Learning.
    Estado: ID del Clúster Semántico (Discreto).
    Acción: Predicción de Sentimiento (0=Negativo, 1=Positivo).
    """
    def __init__(self, dataset_path, n_clusters=10):
        super(SentimentEnv, self).__init__()
        
        # Cargar y preprocesar el dataset
        df = pd.read_csv(dataset_path, sep=";")
        
        # Simular etiquetas binarias para el ejercicio (0 y 1)
        # Asumimos que la columna 'Sentiment' existe o usamos 'Review Rating'
        df['Target'] = np.where(df['Review Rating'] >= 4, 1, 0)
        
        # Vectorización
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        X_tfidf = vectorizer.fit_transform(df["Review Text"].astype(str))
        
        # Discretización del Espacio de Estados usando K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.states = kmeans.fit_predict(X_tfidf)
        self.labels = df['Target'].values
        
        self.n_samples = len(self.labels)
        self.current_step = 0
        
        # Espacio de Acciones (0 o 1) y Espacio de Observaciones (0 a K-1)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(n_clusters)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(0, self.n_samples)
        return self.states[self.current_step], {}

    def step(self, action):
        # Recompensa: +1 si acierta la clasificación, -1 si falla
        correct_label = self.labels[self.current_step]
        reward = 1.0 if action == correct_label else -1.0
        
        # Avanzar al siguiente texto de forma aleatoria
        self.current_step = np.random.randint(0, self.n_samples)
        next_state = self.states[self.current_step]
        
        # En este MDP continuo, nunca terminamos un episodio por defecto
        # pero forzamos el reinicio cada N pasos en el bucle de entrenamiento
        done = False 
        
        return next_state, reward, done, False, {}

# ============================================================
# 2) Entrenamiento del Agente (Q-Learning Tabular)
# ============================================================
N_CLUSTERS = 10
env = SentimentEnv(dataset_path="./data/reviews_sentiment.csv", n_clusters=N_CLUSTERS)

# Inicializar Tabla Q [Estados x Acciones]
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1     # Tasa de aprendizaje
gamma = 0.99    # Factor de descuento
epsilon = 1.0   # Exploración inicial
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 2000
max_steps = 50

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # Política Epsilon-Greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))
            
        next_state, reward, done, _, _ = env.step(action)
        
        # Actualización de Bellman
        td_target = reward + gamma * np.max(Q[next_state])
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error
        
        state = next_state
        total_reward += reward
        
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("\nMatriz Q Optimizada (Estados/Clústeres vs Acciones/Predicciones):")
print(np.round(Q, 2))

