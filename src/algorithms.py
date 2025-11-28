import numpy as np
import random

class KCenterGonzalez:
    """
    Implementação do Algoritmo Guloso (Gonzalez) para o problema k-center.
    
    Este é um algoritmo 2-aproximado. Ele garante que o raio encontrado
    será no máximo 2 vezes o raio da solução ótima.
    
    Lógica:
    1. Escolhe o primeiro centro aleatoriamente.
    2. Escolhe iterativamente o próximo centro como o ponto mais distante
       dos centros já escolhidos (maximizando a distância mínima).
    """

    def __init__(self, k, random_seed=None):
        """
            k (int): Número de clusters/centros a serem encontrados.
            random_seed (int, opcional): Semente para reprodução da escolha do 1º centro.
        """
        self.k = k
        self.centers_indices = [] # Armazena os índices dos pontos escolhidos como centros
        self.radius = 0.0         # Armazena o raio final (distância máxima de cobertura)
        self.labels = []          # Armazena a qual cluster cada ponto pertence
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def fit(self, distance_matrix):
        """
        Executa o algoritmo usando uma matriz de distâncias pré-calculada.

            distance_matrix (np.ndarray): Matriz (N, N) onde M[i,j] é a distância entre i e j.
            
        Retorno:
            tuple: (centers_indices, radius)
        """
        n_samples = distance_matrix.shape[0]
        
        if self.k > n_samples:
            raise ValueError(f"O número de clusters k={self.k} é maior que o número de pontos n={n_samples}.")

        # escolhe do primeiro centro 
        first_center = random.randint(0, n_samples - 1)
        self.centers_indices = [first_center]
        
        # inicializa vetor de distâncias mínimas
        # 'min_dists[i]' guarda a distância do ponto 'i' ao centro mais próximo já escolhido.
        min_dists = distance_matrix[first_center, :].copy()
        
        # loop para escolher os k-1 centros restantes
        for _ in range(1, self.k):
            # O critério guloso: escolher o ponto que está MAIS distante do conjunto atual de centros. Ou seja, o ponto que tem o maior valor em min_dists.
            next_center = np.argmax(min_dists)
            self.centers_indices.append(next_center)
            
            # atualizar as distâncias mínimas
            #  temos um novo centro. 
            
            current_dists_to_new_center = distance_matrix[next_center, :]
            min_dists = np.minimum(min_dists, current_dists_to_new_center)

        # calcular o raio final e atribuições
        
        self.radius = np.max(min_dists)
        
        dist_to_centers = distance_matrix[:, self.centers_indices]
        self.labels = np.argmin(dist_to_centers, axis=1)
        
        return self.centers_indices, self.radius

    def get_metrics_data(self):
        return self.labels