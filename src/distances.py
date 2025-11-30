import numpy as np


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


    pass

def calculate_minkowski_matrix(X, Y, p=2):
    """
    A distancia de Minkowski é definida como:
    D(x, y) = (sum(|x_i - y_i|^p))^(1/p)
    
        X (np.ndarray): Matriz de pontos (N, d).
        Y (np.ndarray): Matriz de centros/pontos (M, d).
        p (int/float): A ordem da distância (p=1: Manhattan, p=2: Euclidiana).
        
    Retorno:
        np.ndarray: Matriz de distâncias (N, M).
    """
    # verificaçao de segurança para dimensões
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Dimensões incompatíveis: X tem {X.shape[1]} dims, Y tem {Y.shape[1]} dims.")

    # criaçao das diferenças usando Broadcasting
    # X[:, np.newaxis, :] transforma X de (N, d) para (N, 1, d)
    # Y[np.newaxis, :, :] transforma Y de (M, d) para (1, M, d)
    # A subtração resulta em um tensor de forma (N, M, d) contendo x_i - y_j para cada coordenada.
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    
    abs_diff = np.abs(diff)
    
    # 3. distância baseada em p
    if p == 1:
        # Manhattan: soma das diferenças absolutas
        # sumariza ao longo do eixo das dimensões (axis=2)
        return np.sum(abs_diff, axis=2)
    
    elif p == 2:
        # Euclidiana: raiz quadrada da soma dos quadrados
        return np.sqrt(np.sum(abs_diff**2, axis=2))
    
    else:
        # caso genérico para p >= 1
        # Soma(|x-y|^p)^(1/p)
        return np.power(np.sum(np.power(abs_diff, p), axis=2), 1.0/p)


def calculate_mahalanobis_matrix(X, Y, VI):
    """
    Calcula a matriz de distâncias de Mahalanobis.
    
    A distância é definida como: D(x, y) = sqrt( (x-y)^T * VI * (x-y) )
    Onde VI é a matriz de covariância inversa.
    
        X (np.ndarray): Matriz de pontos (N, d).
        Y (np.ndarray): Matriz de centros/pontos (M, d).
        VI (np.ndarray): Matriz de covariância inversa (d, d).
        
    Retorno:
        np.ndarray: Matriz de distâncias (N, M).
    """
    # dimensões
    n_samples, n_features = X.shape
    n_centers = Y.shape[0]

    # calcula a diferença entre cada par de pontos (x - y)
    # Resultado shape: (N, M, d)
    delta = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    
    # Para vetorizar a multiplicação matricial (delta * VI * delta^T) de forma eficiente:
    # A fórmula é: soma_sobre_dims( (delta . VI) * delta )
    
    # Redimensionar para 2D temporariamente para usar álgebra linear padrão
    # Transformamos (N, M, d) -> (N*M, d)
    delta_flat = delta.reshape(-1, n_features)
    
    # (x-y)^T * VI
    # equivalente a multiplicar delta_flat por VI
    # Shape: (N*M, d) @ (d, d) -> (N*M, d)
    term1 = np.dot(delta_flat, VI)
    
    # Multiplicar pelo vetor delta original e somar
    # Multiplicação elemento a elemento seguida de soma corresponde ao produto escalar final
    # (x-y)^T * VI * (x-y)
    dist_sq_flat = np.sum(term1 * delta_flat, axis=1)
    
    # redimensionar de volta para a matriz de distâncias (N, M)
    dist_sq = dist_sq_flat.reshape(n_samples, n_centers)
    
    # tratamento de erros numéricos
    dist_sq = np.maximum(dist_sq, 0.0)
    
    return np.sqrt(dist_sq)


def get_covariance_inverse(data):
    """
    Função auxiliar para calcular a inversa da matriz de covariância.
    
        data (np.ndarray): Dados de entrada (N, d) para calcular a covariância.
        
    Retorno:
        np.ndarray: A matriz inversa da covariância (d, d).
    """
    
    # rowvar=False indica que as colunas são as variáveis (dimensões) e linhas são observações
    cov_matrix = np.cov(data, rowvar=False)
    
    # caso especial para 1 dimensão, np.cov retorna array 0-d
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])
        
    # calcular a inversa
    try:
        vi = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Se a matriz for singular (determinante 0), usamos a pseudo-inversa de Moore-Penrose
        print("Aviso: Matriz de covariância singular. Usando pseudo-inversa.")
        vi = np.linalg.pinv(cov_matrix)
        
    return vi
