import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

MAX_SAMPLES = 900 
MIN_SAMPLES = 700

def get_real_datasets():
    datasets = []
    dataset_configs = [
       
        {'id': 4353,  'name': 'UCI_Concrete',         'mode': 'regr'},  # ~1030 
        {'id': 1473,  'name': 'UCI_Energy',           'mode': 'regr'},  # ~768
        {'id': 43926, 'name': 'UCI_Cervical_Cancer',  'mode': 'class'}, # ~858
        {'id': 1510,  'name': 'UCI_Vehicle',          'mode': 'class'}, # ~846
        {'id': 37,    'name': 'UCI_Diabetes',         'mode': 'class'}, # ~768
        {'id': 1464,  'name': 'UCI_Blood_Transfus',   'mode': 'class'}, # ~748
        {'id': 1444,  'name': 'UCI_Annealing',        'mode': 'class'}, # ~798
        {'id': 310,   'name': 'UCI_Mammographic',     'mode': 'class'}, # ~961 
        {'id': 31,    'name': 'UCI_Credit_G',         'mode': 'class'}, # ~1000 
        {'id': 1494,  'name': 'UCI_QSAR',             'mode': 'class'}, # ~1055 
    ]

    print(f"Carregando {len(dataset_configs)} datasets reais da UCI (Max {MAX_SAMPLES} linhas)...")

    for config in dataset_configs:
        try:
            print(f"  -> Baixando {config['name']}...", end=" ")
            data = fetch_openml(data_id=config['id'], as_frame=True, parser='auto')
            df = data.data
            target = data.target
            
            # 1. tratar regressão -> classificação
            if config['mode'] == 'regr':
                if isinstance(target, pd.DataFrame):
                    y_raw = target.iloc[:, 0].values
                else:
                    y_raw = target.values
                # 3 classes para regressão
                y = pd.qcut(y_raw, q=3, labels=False, duplicates='drop').astype(int)
                k = 3
            else:
                # classificação
                if isinstance(target, pd.DataFrame): target = target.iloc[:, 0]
                
                # limpeza básica
                mask = ~pd.isnull(target)
                target = target[mask]
                df = df[mask]
                
                y = LabelEncoder().fit_transform(target)
                k = len(np.unique(y))
                
                # Segurança: se tiver classes demais (>15), reduz para 5 bins
                if k > 15:
                    y = pd.qcut(y, q=5, labels=False, duplicates='drop').astype(int)
                    k = 5

            # 2. tratar features numéricas
            df_numeric = df.select_dtypes(include=[np.number])
    
            X = SimpleImputer(strategy='mean').fit_transform(df_numeric)
            
            # Filtro Mínimo
            if len(X) < MIN_SAMPLES:
                print(f"Ignorado (<{MIN_SAMPLES}).")
                continue
                
            # 3. CORTE DE SEGURANÇA (Max 950)
            if len(X) > MAX_SAMPLES:
                 print(f"[Crt {len(X)}->{MAX_SAMPLES}]", end=" ")
                 indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
                 X = X[indices]
                 y = y[indices]
            
            # Normalização 
            X = StandardScaler().fit_transform(X)
            
            datasets.append({'name': config['name'], 'type': 'real', 'X': X, 'y_true': y, 'k': k})
            print(f"OK (k={k}, n={len(X)}).")
            
        except Exception as e:
            print(f"Erro: {e}")

    return datasets


def get_synthetic_sklearn_shapes(n_samples=750): 
    # n_samples fixo em 750 (dentro da faixa 700-950)
    datasets = []
    print("\nGerando 30 datasets sintéticos (Shapes)...")
    
    for i in range(5):
        seed = 42 + i 
        np.random.seed(seed)
        noise = np.random.uniform(0.05, 0.12)
        
        # 1. Circles
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=noise, random_state=seed)
        datasets.append({'name': f"Syn_Circles_v{i}", 'type': 'syn_shape', 'X': X, 'y_true': y, 'k': 2})

        # 2. Moons
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        datasets.append({'name': f"Syn_Moons_v{i}", 'type': 'syn_shape', 'X': X, 'y_true': y, 'k': 2})

        # 3. Blobs
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise*10, random_state=seed)
        datasets.append({'name': f"Syn_Blobs_v{i}", 'type': 'syn_shape', 'X': X, 'y_true': y, 'k': 3})

        # 4. Anisotrópicos
        X_base, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=seed)
        transform = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X_base, transform)
        datasets.append({'name': f"Syn_Aniso_v{i}", 'type': 'syn_shape', 'X': X, 'y_true': y, 'k': 3})

        # 5. Variância Variada
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
        datasets.append({'name': f"Syn_VarVar_v{i}", 'type': 'syn_shape', 'X': X, 'y_true': y, 'k': 3})

        # 6. Sem Estrutura
        X = np.random.rand(n_samples, 2)
        y = np.zeros(n_samples, dtype=int)
        datasets.append({'name': f"Syn_NoStruct_v{i}", 'type': 'syn_shape', 'X': X, 'y_true': y, 'k': 3})

    return datasets


def get_synthetic_multivariate(n_samples=750):
    # n_samples fixo em 750
    datasets = []
    print("\nGerando 10 datasets sintéticos (Multivariado)...")
    
    n_centers = 3
    pts = n_samples // n_centers
    centers_fixed = np.array([[0, 10], [10, 0], [-5, -5]])

    # A. Variando Sobreposição
    for i, std in enumerate([0.5, 1.5, 2.5, 3.5, 5.0]):
        X_l, y_l = [], []
        for c in range(n_centers):
            cov = np.eye(2) * std
            X_l.append(np.random.multivariate_normal(centers_fixed[c], cov, pts))
            y_l.append(np.full(pts, c))
        datasets.append({'name': f"Syn_Overlap_Lv{i+1}", 'type': 'syn_multi', 
                         'X': np.vstack(X_l), 'y_true': np.hstack(y_l), 'k': n_centers})

    # B. Variando Elongação
    for i, el in enumerate([1, 5, 15, 30, 60]):
        X_l, y_l = [], []
        for c in range(n_centers):
            base = np.array([[el, 0], [0, 1]])
            th = np.radians(np.random.uniform(0, 360))
            rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            cov = rot @ base @ rot.T
            X_l.append(np.random.multivariate_normal(centers_fixed[c], cov, pts))
            y_l.append(np.full(pts, c))
        datasets.append({'name': f"Syn_Elong_Lv{i+1}", 'type': 'syn_multi', 
                         'X': np.vstack(X_l), 'y_true': np.hstack(y_l), 'k': n_centers})
        
    return datasets

def get_all_datasets():
    return get_real_datasets() + get_synthetic_sklearn_shapes() + get_synthetic_multivariate()