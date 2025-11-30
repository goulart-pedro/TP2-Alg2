import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.distances import calculate_minkowski_matrix, calculate_mahalanobis_matrix, get_covariance_inverse
from src.algorithms import KCenterGonzalez, KCenterRefined

class ExperimentRunner:
    def __init__(self, output_file="relatorio_final_tp2.csv"):
        self.results = []
        self.output_file = output_file

    def _log(self, dataset, algo, metric, param, radius_mean, radius_std, time_mean, ari_mean, sil_mean):
        """Helper para adicionar uma linha aos resultados."""
        entry = {
            'Dataset': dataset,
            'Algoritmo': algo,
            'Distancia': metric,
            'Parametro': param,  # Ex: "Largura=1%" ou "Seed=Media"
            'Raio_Mean': round(radius_mean, 4),
            'Raio_Std': round(radius_std, 4),
            'Tempo_Mean': round(time_mean, 6),
            'ARI': round(ari_mean, 4),
            'Silhueta': round(sil_mean, 4)
        }
        self.results.append(entry)
        # Salva parcial a cada iteração para não perder dados se travar
        pd.DataFrame(self.results).to_csv(self.output_file, index=False)

    def run_benchmark(self, dataset_name, X, y_true, k):
        """
        Roda a bateria completa de testes para um dataset específico.
        """
        # definir as configurações de distância a testar
        # Lista de tuplas: (Nome, Função que recebe X e retorna Matriz D)
        dist_configs = [
            ('Manhattan', lambda d: calculate_minkowski_matrix(d, d, p=1)),
            ('Euclidiana', lambda d: calculate_minkowski_matrix(d, d, p=2))
        ]

        # Tenta adicionar Mahalanobis (pode falhar se matriz for singular demais)
        try:
            VI = get_covariance_inverse(X)
            dist_configs.append(('Mahalanobis', lambda d: calculate_mahalanobis_matrix(d, d, VI)))
        except Exception as e:
            print(f"  [Aviso] Mahalanobis pulado para {dataset_name}: {e}")

        for dist_name, dist_func in dist_configs:
            # O PDF exige: calcular matriz UMA VEZ
            start_t = time.time()
            D = dist_func(X)
            mat_time = time.time() - start_t
            
            # gonzales
            raios, tempos, aris, sils = [], [], [], []
            
            for i in range(15):
                # Seed varia de 0 a 14
                model = KCenterGonzalez(k, random_seed=i)
                
                t0 = time.time()
                _, r = model.fit(D) # Fit usando a matriz pré-calculada
                tf = time.time()
                
                raios.append(r)
                tempos.append(tf - t0) # Não somamos mat_time aqui pois a matriz é input
                
                # Métricas de Qualidade
                aris.append(adjusted_rand_score(y_true, model.labels))
                try:
                    # Silhueta pode falhar se k=1 ou labels unicos
                    if len(set(model.labels)) > 1:
                        sils.append(silhouette_score(X, model.labels))
                    else:
                        sils.append(-1)
                except: sils.append(-1)

            # Loga a média das 15 execuções
            self._log(dataset_name, 'Gonzalez', dist_name, '15_Runs_Avg', 
                      np.mean(raios), np.std(raios), np.mean(tempos), 
                      np.mean(aris), np.mean(sils))


            # refinamento
            max_dist = np.max(D)
            widths_pct = [0.01, 0.05, 0.10, 0.25] # 1%, 5%, 10%, 25%
            
            for pct in widths_pct:
                target_width = max_dist * pct
                model = KCenterRefined(k, min_width=target_width)
                
                t0 = time.time()
                _, r = model.fit(D)
                tf = time.time()
                
                # Refinement foca em Raio, mas calculamos ARI se possível
                ari = adjusted_rand_score(y_true, model.labels)
                try: sil = silhouette_score(X, model.labels)
                except: sil = -1
                
                self._log(dataset_name, 'Refinement', dist_name, f'Width_{int(pct*100)}%',
                          r, 0.0, tf - t0, ari, sil)

        # k-means
        # K-Means usa Euclidiana por padrão. Rodamos fora do loop de distâncias.
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        
        t0 = time.time()
        labels_km = km.fit_predict(X)
        tf = time.time()
    
        centers = km.cluster_centers_
        d_to_centers = calculate_minkowski_matrix(X, centers, p=2)
        # Pega distancia pro centro mais proximo
        min_dists = np.min(d_to_centers, axis=1)
        raio_km = np.max(min_dists)
        
        ari_km = adjusted_rand_score(y_true, labels_km)
        try: sil_km = silhouette_score(X, labels_km)
        except: sil_km = -1
        
        self._log(dataset_name, 'KMeans_Sklearn', 'Euclidiana', 'Default',
                  raio_km, 0.0, tf - t0, ari_km, sil_km)