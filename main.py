import matplotlib.pyplot as plt
import numpy as np
import sys
from src.distances import calculate_minkowski_matrix
from src.algorithms import KCenterGonzalez

# Gerar 100 pontos aleatórios
np.random.seed(2)  # Para resultados reproduzíveis
x = np.random.rand(100)
y = np.random.rand(100)

# Criar o gráfico
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7, color='blue')
plt.title('Pontos Aleatórios (x, y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)

k = 2
points = list(zip(x, y))  # Convertendo para lista

# inicializa k centroides de duas coordenadas
centroids = np.random.rand(k, 2)  # matriz k x 2
bins = [[] for i in range(k)]

# Plotar os centroides iniciais
for i in range(k):
    plt.scatter(centroids[i, 0], centroids[i, 1], color='red', marker='x', s=100, linewidth=3)

# separar os pontos
for point in points:
    from numpy.linalg import norm
    
    # seleciona o centroide cuja distancia até o ponto é mínima
    def comp_dist(centroid):
        return norm(np.array(point) - np.array(centroid))
    
    # Encontrar o índice do centróide mais próximo
    distances = [comp_dist(centroid) for centroid in centroids]
    assigned_centroid_idx = np.argmin(distances)
    
    # Adicionar ponto ao bin correspondente
    bins[assigned_centroid_idx].append(point)

# Plotar os pontos coloridos por cluster
colors = ['blue', 'magenta']
for i in range(k):
    if bins[i]:  # Verifica se o bin não está vazio
        cluster_points = np.array(bins[i])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   alpha=0.7, color=colors[i], label=f'Cluster {i+1}')

# Plotar os centroides finais novamente para destacar
for i in range(k):
    plt.scatter(centroids[i, 0], centroids[i, 1], color='red', marker='x', 
               s=100, linewidth=3, label='Centróides' if i == 0 else "")

plt.legend()
plt.savefig('kmeans_clusters.png')
plt.show()

# Mostrar estatísticas
print(f"Total de pontos: {len(points)}")
for i in range(k):
    print(f"Cluster {i+1}: {len(bins[i])} pontos")
    
print("\n" + "="*50)
print("INICIANDO EXECUÇÃO DO ALGORITMO DE GONZALEZ (TP2)")
print("="*50)

try:
    from src.distances import calculate_minkowski_matrix
    from src.algorithms import KCenterGonzalez
except ImportError:
    print("ERRO: Não foi possível importar 'src'. Verifique se a pasta e arquivos existem.")
    sys.exit(1)


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


