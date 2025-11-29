import matplotlib.pyplot as plt
import numpy as np
import sys
from src.distances import calculate_minkowski_matrix
from src.algorithms import KCenterGonzalez, KCenterGreedy

POINT_AMOUNT = 100

# Gerar 100 pontos aleatórios
np.random.seed(2)  # Para resultados reproduzíveis
x = np.random.rand(POINT_AMOUNT)
y = np.random.rand(POINT_AMOUNT)

print("\n" + "="*50)
print("INICIANDO EXECUÇÃO DO ALGORITMO DE GONZALEZ (TP2)")
print("="*50)

data_tp2 = np.column_stack((x, y))

print("Calculando matriz de distâncias (Euclidiana)...")
dist_matrix = calculate_minkowski_matrix(data_tp2, data_tp2, p=2) 

k_gonzalez = 3  
print(f"Executando Gonzalez para k={k_gonzalez}...")

gonzalez_algo = KCenterGonzalez(k=k_gonzalez, random_seed=42)
centros_idx, raio = gonzalez_algo.fit(dist_matrix)

print(f"-> Raio Final (Custo): {raio:.4f}")
print(f"-> Índices dos Centros Escolhidos: {centros_idx}")
print(f"-> Coordenadas dos Centros:\n{data_tp2[centros_idx]}")

# plotar Resultado do Gonzalez
plt.figure(figsize=(8, 6))

# plota todos os pontos em cinza
plt.scatter(data_tp2[:, 0], data_tp2[:, 1], c='gray', alpha=0.6, label='Pontos')

# plota os centros escolhidos em vermelho 
centros_coords = data_tp2[centros_idx]
plt.scatter(centros_coords[:, 0], centros_coords[:, 1], c='red', s=200, marker='*', label='Centros (Gonzalez)')

# desenha círculos representando o raio de cobertura 
ax = plt.gca()
for center in centros_coords:
    circle = plt.Circle((center[0], center[1]), raio, color='red', fill=False, linestyle='--', alpha=0.3)
    ax.add_patch(circle)

plt.title(f'Resultado Gonzalez (k={k_gonzalez}) - Raio: {raio:.4f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

print("Exibindo gráfico Gonzalez (TP2)...")
plt.show()
plt.savefig('image.svg')


