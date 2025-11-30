import matplotlib.pyplot as plt
import numpy as np
import sys
from src.distances import calculate_minkowski_matrix
from src.algorithms import KCenterGonzalez, KCenterRefinement
from typing import Union

class Metric:
    """Classe para cálculo de métricas de distância entre pontos."""
    
    @staticmethod
    def euclidean(x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    
    @staticmethod
    def manhattan(x, y):
        return np.sum(np.abs(x - y))
    
    @staticmethod
    def minkowski(x, y, p=2):
        return np.sum(np.abs(x - y) ** p) ** (1/p)
    
    @staticmethod
    def get_distance_function(metric):
        """Retorna a função de distância apropriada."""
        if metric == 'euclidean':
            return Metric.euclidean
        elif metric == 'manhattan':
            return Metric.manhattan
        elif metric == 'minkowski':
            return Metric.minkowski
        else:
            raise ValueError(f"Métrica não suportada: {metric}")

class KCenter:
    def __init__(self, data_tp2, k, metric='euclidean'):
        self.data_tp2 = data_tp2
        self.k = k
        self.distance_matrix = None
        self.metric = metric

    def inertia(self, center_coordinates, labels):
        """
        Calcula a inércia (Within-Cluster Sum of Squares) para o agrupamento.
        
        Args:
            center_coordinates: Array com as coordenadas dos centros (k x d)
            labels: Array com as atribuições de cluster para cada ponto (n)
            
        Returns:
            float: Valor da inércia (soma das distâncias quadradas aos centros)
        """
        total_inertia = 0.0
        distance_func = Metric.get_distance_function(self.metric)
        
        for cluster_id in range(self.k):
            # Encontra todos os pontos pertencentes a este cluster
            cluster_mask = (labels == cluster_id)
            cluster_points = self.data_tp2[cluster_mask]
            
            if len(cluster_points) > 0:
                # Centro deste cluster
                center = center_coordinates[cluster_id]
                
                # Calcula distâncias quadradas do centro para todos os pontos do cluster
                for point in cluster_points:
                    distance = distance_func(point, center)
                    total_inertia += distance ** 2
        
        return total_inertia

    def fit(self, method_instance):
        """
        Executa o algoritmo k-center usando uma instância do método já criada.
        
        Args:
            method_instance: Instância já criada do algoritmo (ex: KCenterGreedy(k=5))
            
        Returns:
            tuple: (centers_indices, radius, labels)
        """
        if self.distance_matrix is None:
            self._compute_distance_matrix(self.metric)
        
        # Usa a instância do método passada como parâmetro
        centers_indices, radius = method_instance.fit(self.distance_matrix)
        
        return centers_indices, radius, method_instance.labels

    
    def _compute_distance_matrix(self, metric: str = 'euclidean'):
        """
        Calcula a matriz de distância entre todos os pares de pontos.
        """
        n_samples = self.data_tp2.shape[0]
        self.distance_matrix = np.zeros((n_samples, n_samples))
        distance_func = Metric.get_distance_function(metric)
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = distance_func(self.data_tp2[i], self.data_tp2[j])
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist

POINT_AMOUNT = 100
k = 3
# Gerar 100 pontos aleatórios
np.random.seed(42)  # Para resultados reproduzíveis
x = np.random.rand(POINT_AMOUNT)
y = np.random.rand(POINT_AMOUNT)

data_tp2 = np.column_stack((x, y))
kcenter = KCenter(data_tp2, k, metric='euclidean')

# não acho que deveria haver a necessidade de passar k aqui
# tampouco a seed (opcional)
# isso deveria ser gerenciado pela classe KCenter
method_name = 'refinement'

if method_name == 'gonzalez':
    method = KCenterGonzalez(k, 42)
elif method_name == 'refinement':
    method = KCenterRefinement(k)

print("\n" + "="*50)
print(f"INICIANDO EXECUÇÃO DO ALGORITMO ({method_name.capitalize()}) (TP2)")
print("="*50)

print(f"Executando para k={kcenter.k} ({method_name.capitalize()})...")
centros_idx, raio, labels = kcenter.fit(method)

print(f"-> Raio Final (Custo): {raio:.4f}")
print(f"-> Índices dos Centros Escolhidos: {centros_idx}")
print(f"-> Coordenadas dos Centros:\n{data_tp2[centros_idx]}")

# Calcular inércia usando a nova classe Metric
centros_coords = data_tp2[centros_idx]
inercia = kcenter.inertia(centros_coords, labels)
print(f"-> Inércia: {inercia:.4f}")

# plotar Resultado da execução
plt.figure(figsize=(8, 6))

# plota todos os pontos em cinza
plt.scatter(data_tp2[:, 0], data_tp2[:, 1], c='gray', alpha=0.6, label='Pontos')

# plota os centros escolhidos em vermelho 
plt.scatter(centros_coords[:, 0], centros_coords[:, 1], c='red', s=200, marker='*', label=f'Centros ({method_name.capitalize()})')

# desenha círculos representando o raio de cobertura 
ax = plt.gca()
for center in centros_coords:
    circle = plt.Circle((center[0], center[1]), raio, color='red', fill=False, linestyle='--', alpha=0.3)
    ax.add_patch(circle)

plt.title(f'Resultado {method_name.capitalize()} (k={k}) - Raio: {raio:.4f} - Inércia: {inercia:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

print("Exibindo gráfico...")
plt.show()
plt.savefig('image.svg')
