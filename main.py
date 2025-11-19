import matplotlib.pyplot as plt
import numpy as np

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
