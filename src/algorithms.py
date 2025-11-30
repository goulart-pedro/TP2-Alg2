import numpy as np
import random

class KCenterGonzalez:
    """
    Implementação do Algoritmo Guloso (Gonzalez) para o problema k-center.
    """

    def __init__(self, k, random_seed=None):
        self.k = k
        self.centers_indices = []
        self.radius = 0.0
        self.labels = []
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def fit(self, distance_matrix):
        """
        Executa o algoritmo Gonzalez (GULOSO).
        """
        n_samples = distance_matrix.shape[0]
        
        if self.k > n_samples:
            raise ValueError(f"k={self.k} é maior que o número de pontos.")

        # 1. Escolhe o primeiro centro aleatoriamente
        first_center = random.randint(0, n_samples - 1)
        self.centers_indices = [first_center]
        
        # Inicializa distâncias mínimas
        min_dists = distance_matrix[first_center, :].copy()
        
        # 2. Escolhe os demais k-1 centros (Lógica Gulosa)
        for _ in range(1, self.k):
            # O critério guloso: escolher o ponto mais distante dos centros atuais
            next_center = np.argmax(min_dists)
            self.centers_indices.append(next_center)
            
            # Atualizar as distâncias mínimas
            min_dists = np.minimum(min_dists, distance_matrix[next_center, :])

        # 3. Recalcular Raio e Labels
        dist_to_centers = distance_matrix[:, self.centers_indices]
        self.labels = np.argmin(dist_to_centers, axis=1)
        
        # O raio é a distância máxima do ponto ao seu centro
        self.radius = np.max(np.min(dist_to_centers, axis=1))
        
        return self.centers_indices, self.radius

    def get_metrics_data(self):
        return self.labels


class KCenterRefined:  
    """
    Algoritmo de refinamento de intervalos para k-center (Busca Binária).
    """
    
    def __init__(self, k, min_width=0.01):
        self.k = k
        self.min_width = min_width
        self.centers_indices = []
        self.radius = 0.0
        self.labels = []
    
    def fit(self, distance_matrix):
        """
        Executa a Busca Binária no raio.
        """
        n_samples = distance_matrix.shape[0]
        
        # Intervalo de busca [0, max_dist]
        low = 0.0
        high = np.max(distance_matrix)
        
        best_centers = []
        
        # Busca Binária
        while (high - low) > self.min_width:
            mid = (low + high) / 2
            
            # Verifica se cobre
            centers = self._can_cover(distance_matrix, mid)
            
            if centers is not None:
                best_centers = centers
                high = mid # Tenta raio menor
            else:
                low = mid # Precisa de raio maior
        
        # Fallback
        if not best_centers:
            best_centers = list(range(self.k))

        self.centers_indices = best_centers
        
        # Calcular raio real final
        dist_to_centers = distance_matrix[:, self.centers_indices]
        self.labels = np.argmin(dist_to_centers, axis=1)
        self.radius = np.max(np.min(dist_to_centers, axis=1))
        
        return self.centers_indices, self.radius
    
    def _can_cover(self, distance_matrix, r):
        """
        Verifica cobertura gulosa com raio r.
        """
        n_samples = distance_matrix.shape[0]
        uncovered = set(range(n_samples))
        centers = []
        
        while len(centers) < self.k and uncovered:
            u = next(iter(uncovered))
            centers.append(u)
            
            # Remove pontos cobertos por 2*r
            covered_indices = np.where(distance_matrix[u, :] <= 2 * r)[0]
            uncovered.difference_update(covered_indices)
        
        if len(uncovered) == 0:
            return centers
        return None