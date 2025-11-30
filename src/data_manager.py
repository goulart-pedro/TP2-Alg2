import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

MAX_SAMPLES = 900  
MIN_SAMPLES = 700  

def safe_load_openml(dataset_id, name, mode):
    """
    Função blindada para carregar e limpar datasets da UCI.
    """
    try:
        print(f"  -> Baixando {name}...", end=" ")
        X_raw, y_raw = fetch_openml(data_id=dataset_id, as_frame=True, return_X_y=True, parser='auto')
        
        if isinstance(y_raw, pd.DataFrame):
            y_raw = y_raw.iloc[:, 0]
        
        # erros viram NaN
        y_numeric = pd.to_numeric(y_raw, errors='coerce')
        
        # Lógica de Binning 
        num_unique = y_raw.nunique()
        
        if mode == 'regr' or num_unique > 20:
            valid_idx = ~y_numeric.isna()
            X_raw = X_raw[valid_idx]
            y_numeric = y_numeric[valid_idx]
            
            try:
                y = pd.qcut(y_numeric, q=3, labels=False, duplicates='drop').astype(int)
            except:
                median = y_numeric.median()
                y = (y_numeric > median).astype(int)
        else:
            valid_idx = ~y_raw.isna()
            X_raw = X_raw[valid_idx]
            y_raw = y_raw[valid_idx]
            
            le = LabelEncoder()
            y = le.fit_transform(y_raw)

        k = len(np.unique(y))
        
        X_df = X_raw.apply(pd.to_numeric, errors='coerce')
        
        # Drop colunas que ficaram inteiramente vazias 
        X_df = X_df.dropna(axis=1, how='all')
        
        # Preenche NaNs restantes com a média
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X_df)
        
        # filtro
        if len(X) < MIN_SAMPLES:
            print(f"Ignorado (Muito pequeno: {len(X)}).")
            return None

        if len(X) > MAX_SAMPLES:
            indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
            X = X[indices]
            y = y[indices]
        
      
        if k >= len(X):
            y = np.random.randint(0, 3, size=len(X)) # Classes dummy
            k = 3
        
        # normalização
        X = StandardScaler().fit_transform(X)
        
        print(f"OK (k={k}, n={len(X)}).")
        return {'name': name, 'type': 'real', 'X': X, 'y_true': y, 'k': k}

    except Exception as e:
        print(f"Erro: {e}")
        return None

def get_real_datasets():
    print(f"Carregando datasets reais (Max {MAX_SAMPLES} linhas)...")
    
    configs = [
        {'id': 4353,  'name': 'UCI_Concrete',         'mode': 'regr'}, 
        {'id': 1473,  'name': 'UCI_Energy',           'mode': 'regr'}, 
        {'id': 43926, 'name': 'UCI_Cervical_Cancer',  'mode': 'regr'}, 
        {'id': 1510,  'name': 'UCI_Vehicle',          'mode': 'class'},
        {'id': 37,    'name': 'UCI_Diabetes',         'mode': 'class'},
        {'id': 1464,  'name': 'UCI_Blood_Transfus',   'mode': 'class'},
        {'id': 1444,  'name': 'UCI_Annealing',        'mode': 'class'},
        {'id': 310,   'name': 'UCI_Mammographic',     'mode': 'class'},
        {'id': 1494,  'name': 'UCI_QSAR',             'mode': 'class'}, 
        {'id': 60,    'name': 'UCI_Waveform',         'mode': 'class'},
    ]

    results = []
    for cfg in configs:
        res = safe_load_openml(cfg['id'], cfg['name'], cfg['mode'])
        if res:
            results.append(res)
            
    return results

#sinteticos
def get_synthetic_sklearn_shapes(n_samples=750):
    datasets = []
    print("\nGerando 30 datasets sintéticos (Shapes)...")
    for i in range(5):
        s = 42 + i; np.random.seed(s); n = np.random.uniform(0.05, 0.12)
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=n, random_state=s)
        datasets.append({'name': f"Syn_Circ_{i}", 'type': 'syn', 'X': X, 'y_true': y, 'k': 2})
        X, y = make_moons(n_samples=n_samples, noise=n, random_state=s)
        datasets.append({'name': f"Syn_Moon_{i}", 'type': 'syn', 'X': X, 'y_true': y, 'k': 2})
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=n*10, random_state=s)
        datasets.append({'name': f"Syn_Blob_{i}", 'type': 'syn', 'X': X, 'y_true': y, 'k': 3})
        Xb, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=s)
        X = np.dot(Xb, [[0.6, -0.6], [-0.4, 0.8]])
        datasets.append({'name': f"Syn_Anis_{i}", 'type': 'syn', 'X': X, 'y_true': y, 'k': 3})
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=s)
        datasets.append({'name': f"Syn_Var_{i}", 'type': 'syn', 'X': X, 'y_true': y, 'k': 3})
        X = np.random.rand(n_samples, 2); y = np.zeros(n_samples, int)
        datasets.append({'name': f"Syn_Rand_{i}", 'type': 'syn', 'X': X, 'y_true': y, 'k': 3}) # k=3 forçado
    return datasets

def get_synthetic_multivariate(n_samples=750):
    datasets = []
    print("\nGerando 10 datasets sintéticos (Multivariado)...")
    cents = np.array([[0, 10], [10, 0], [-5, -5]])
    pts = n_samples // 3
    for i, std in enumerate([0.5, 1.5, 2.5, 3.5, 5.0]):
        X_l, y_l = [], []
        for c in range(3):
            X_l.append(np.random.multivariate_normal(cents[c], np.eye(2)*std, pts))
            y_l.append(np.full(pts, c))
        datasets.append({'name': f"Syn_Over_{i}", 'type': 'syn', 'X': np.vstack(X_l), 'y_true': np.hstack(y_l), 'k': 3})
    for i, el in enumerate([1, 5, 15, 30, 60]):
        X_l, y_l = [], []
        for c in range(3):
            th = np.radians(np.random.uniform(0, 360)); rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            cov = rot @ np.array([[el, 0], [0, 1]]) @ rot.T
            X_l.append(np.random.multivariate_normal(cents[c], cov, pts))
            y_l.append(np.full(pts, c))
        datasets.append({'name': f"Syn_Elon_{i}", 'type': 'syn', 'X': np.vstack(X_l), 'y_true': np.hstack(y_l), 'k': 3})
    return datasets

def get_all_datasets():
    return get_real_datasets() + get_synthetic_sklearn_shapes() + get_synthetic_multivariate()