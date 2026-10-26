import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. Cargar los datos de los dos entrenamientos
    try:
        q_rewards = np.load("q_learning_rewards.npy")
        dqn_rewards = np.load("dqn_rewards.npy")
    except FileNotFoundError as e:
        print("Error: No se encontraron los archivos de datos.")
        print("Asegúrate de ejecutar primero los scripts de Q-Learning y DQN.")
        return

    # 2. Función de media móvil para suavizar las curvas y ver la tendencia
    def moving_average(data, window_size=30):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # 3. Configuración de la gráfica comparativa
    plt.figure(figsize=(10, 6))

    # Trazar la curva de Q-Learning Clásico (naranja)
    plt.plot(moving_average(q_rewards), 
             label="Q-Learning (Estados discretizados)", 
             color="#FF8C00", linewidth=2)

    # Trazar la curva de Deep Q-Network (azul)
    plt.plot(moving_average(dqn_rewards), 
             label="DQN (Red Neuronal Profunda)", 
             color="#1E90FF", linewidth=2)

    # Línea de referencia del éxito en CartPole (475 puntos)
    plt.axhline(y=475, color='red', linestyle='--', alpha=0.5, 
                label='Umbral de éxito (475)')

    # Detalles de formato
    plt.title("Análisis Comparativo de Estabilidad: Q-Learning vs DQN en CartPole-v1", fontsize=13, pad=15)
    plt.xlabel("Episodios", fontsize=11)
    plt.ylabel("Recompensa Media Móvil (30 episodios)", fontsize=11)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # 4. Exportar la imagen correctamente para el reporte
    os.makedirs('resultados', exist_ok=True)
    ruta_guardado = "resultados/U2-2_Grafica_Comparativa_Agentes.png"
    
    # bbox_inches='tight' evita que se corten los bordes o salgan los iconos de la ventana
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"Gráfica comparativa exportada exitosamente en: {ruta_guardado}")
    
    # Mostrar la gráfica en pantalla
    plt.show()

if __name__ == "__main__":
    main()