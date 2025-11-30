import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.distances import calculate_minkowski_matrix, calculate_mahalanobis_matrix, get_covariance_inverse
from src.algorithms import KCenterGonzalez, KCenterRefined
from src.data_manager import get_all_datasets
from src.experiments import ExperimentRunner
from src.plotter import gerar_graficos_e_tabelas 

class KCenter:
    def __init__(self, data_tp2, k, metric='euclidean'):
        self.data_tp2 = data_tp2
        self.k = k
        self.metric = metric
        self.distance_matrix = None
        self.labels = None

    def fit(self, method_instance):
        if self.distance_matrix is None:
            self._compute_distance_matrix(self.metric)
        
        centers_indices, radius = method_instance.fit(self.distance_matrix)
        
        if hasattr(method_instance, 'labels'):
            self.labels = method_instance.labels
        else:
            dists = self.distance_matrix[:, centers_indices]
            self.labels = np.argmin(dists, axis=1)

        return centers_indices, radius, self.labels

    def _compute_distance_matrix(self, metric: str):
        if metric == 'euclidean':
            self.distance_matrix = calculate_minkowski_matrix(self.data_tp2, self.data_tp2, p=2)
        elif metric == 'manhattan':
            self.distance_matrix = calculate_minkowski_matrix(self.data_tp2, self.data_tp2, p=1)
        elif metric == 'minkowski':
            self.distance_matrix = calculate_minkowski_matrix(self.data_tp2, self.data_tp2, p=3)
        elif metric == 'mahalanobis':
            VI = get_covariance_inverse(self.data_tp2)
            self.distance_matrix = calculate_mahalanobis_matrix(self.data_tp2, self.data_tp2, VI)
        else:
            raise ValueError(f"Métrica não suportada: {metric}")

    def inertia(self, center_coordinates, labels):
        if labels is None: return 0.0
        total_inertia = 0.0
        for cluster_id in range(self.k):
            mask = (labels == cluster_id)
            points = self.data_tp2[mask]
            if len(points) > 0:
                center = center_coordinates[cluster_id]
                dists = np.sum((points - center) ** 2, axis=1)
                total_inertia += np.sum(dists)
        return total_inertia

# demo visual
def run_original_demo():
    POINT_AMOUNT = 100
    k = 3
    
    np.random.seed(42)
    x = np.random.rand(POINT_AMOUNT)
    y = np.random.rand(POINT_AMOUNT)

    data_tp2 = np.column_stack((x, y))
    kcenter = KCenter(data_tp2, k, metric='euclidean')

    print("\n--- Configuração ---")
    print("1. Gonzalez")
    print("2. Refinement")
    opt = input("Escolha o método (1 ou 2): ")
    
    if opt == '2':
        method_name = 'refinement'
        method = KCenterRefined(k, min_width=0.01)
    else:
        method_name = 'gonzalez'
        method = KCenterGonzalez(k, random_seed=42)

    print(f"\nINICIANDO EXECUÇÃO ({method_name.capitalize()})...")
    centros_idx, raio, labels = kcenter.fit(method)

    print(f"-> Raio Final: {raio:.4f}")
    centros_coords = data_tp2[centros_idx]
    inercia = kcenter.inertia(centros_coords, labels)
    print(f"-> Inércia: {inercia:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(data_tp2[:, 0], data_tp2[:, 1], c='gray', alpha=0.6, label='Pontos')
    plt.scatter(centros_coords[:, 0], centros_coords[:, 1], c='red', s=200, marker='*', label='Centros')

    ax = plt.gca()
    for center in centros_coords:
        circle = plt.Circle((center[0], center[1]), raio, color='red', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(circle)

    plt.title(f'{method_name.capitalize()} (k={k}) - Raio: {raio:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('image.svg')
    print("Exibindo gráfico...")
    plt.show()

# experimentos
def run_full_report():
    print("\n" + "="*60)
    print(" GERANDO DADOS PARA O RELATÓRIO (UCI + Sintéticos)")
    print("="*60)
    datasets = get_all_datasets()
    runner = ExperimentRunner("relatorio_final_tp2.csv")
    
    for i, data in enumerate(datasets):
        print(f"[{i+1}/{len(datasets)}] Processando {data['name']}...")
        runner.run_benchmark(data['name'], data['X'], data['y_true'], data['k'])
        
    print("\nSucesso! Arquivo 'relatorio_final_tp2.csv' criado.")

#gerar graficos
def run_data_analysis():
    print("\n" + "="*60)
    print(" GERANDO GRÁFICOS E TABELAS (A partir do CSV)")
    print("="*60)
    gerar_graficos_e_tabelas("relatorio_final_tp2.csv")

# menu
if __name__ == "__main__":
    while True:
        print("\n=== TP2 - K-Center ===")
        print("1. Demo Visual (Gráfico Simples)")
        print("2. Rodar Experimentos (Gera CSV)")
        print("3. Analisar Dados (Gera PNGs e Tabela)")
        print("0. Sair")
        
        try:
            opt = input("Opção: ")
            if opt == '1':
                run_original_demo()
            elif opt == '2':
                run_full_report()
            elif opt == '3':
                run_data_analysis()
            elif opt == '0':
                break
            else:
                print("Opção inválida.")
        except KeyboardInterrupt:
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"Erro inesperado: {e}")