import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Suprimir warnings de divisão por zero nas médias espectrais (comportamento esperado do script original)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def split_psa_columns(df_psa: pd.DataFrame, train_ids: list, test_ids: list):
    """
    Separa as colunas do DataFrame PSA baseando-se nos IDs de terremoto
    do conjunto de treino e teste.
    """
    train_ids_str = set(str(x) for x in train_ids)
    test_ids_str = set(str(x) for x in test_ids)
    
    train_cols = ['period(s)']
    test_cols = ['period(s)']
    
    for col in df_psa.columns:
        if col == 'period(s)':
            continue
        
        if any(t_id in col for t_id in train_ids_str):
            train_cols.append(col)
        elif any(t_id in col for t_id in test_ids_str):
            test_cols.append(col)
            
    return df_psa[train_cols].copy(), df_psa[test_cols].copy()

def calculate_psa_means(df_psa_subset: pd.DataFrame):
    """
    Processa as matrizes espectrais separando e calculando as médias 
    das direções V, EW e NS por estação sísmica.
    """
    columns = df_psa_subset.columns
    stations = set()
    
    for column in columns:
        if column == 'period(s)':
            continue
        station = column.split('-')[0]
        stations.add(station)

    stations_list = list(stations)
    
    df_spectral_coefs_mean = pd.DataFrame(columns=stations_list)
    df_spectral_mean = pd.DataFrame(columns=stations_list)
    df_horiz_spectral_mean = pd.DataFrame(columns=stations_list)
    df_vertical_spectral_mean = pd.DataFrame(columns=stations_list)
    
    for station_key in stations_list:
        v_signals = []
        ew_signals = []
        ns_signals = []
        
        for i, col_name in enumerate(columns):
            if col_name == 'period(s)':
                continue
                
            actual_station_key = col_name.split("-")[0]
            orientation = col_name.split("-")[-1]
            
            if actual_station_key == station_key:
                signal = df_psa_subset.iloc[:, i].values
                if orientation == 'V':
                    v_signals.append(signal)
                elif orientation == 'EW':
                    ew_signals.append(signal)
                elif orientation == 'NS':
                    ns_signals.append(signal)
        
        # Calcula médias por orientação (axis=0)
        v_mean = np.mean(v_signals, axis=0) if v_signals else np.zeros(len(df_psa_subset))
        ew_mean = np.mean(ew_signals, axis=0) if ew_signals else np.zeros(len(df_psa_subset))
        ns_mean = np.mean(ns_signals, axis=0) if ns_signals else np.zeros(len(df_psa_subset))
        
        # Coeficientes e médias gerais
        ew_v_spectral_coeficients = ew_mean / v_mean
        ns_v_spectral_coeficients = ns_mean / v_mean
        
        spectral_coefs_mean = np.mean(np.array([ew_v_spectral_coeficients, ns_v_spectral_coeficients]), axis=0)
        spectral_means = np.mean(np.array([v_mean, ew_mean, ns_mean]), axis=0)
        h_spectral_means = np.mean(np.array([ew_mean, ns_mean]), axis=0)
        
        # Atribuição aos DataFrames de retorno
        df_spectral_coefs_mean[station_key] = spectral_coefs_mean
        df_spectral_mean[station_key] = spectral_means
        df_horiz_spectral_mean[station_key] = h_spectral_means
        df_vertical_spectral_mean[station_key] = v_mean
        
    return {
        'spectral_coefs': df_spectral_coefs_mean,
        'spectral_means': df_spectral_mean,
        'spectral_horiz_means': df_horiz_spectral_mean,
        'spectral_vert_means': df_vertical_spectral_mean
    }

def generate_clusters(df_train_mean: pd.DataFrame, df_test_mean: pd.DataFrame, k: int = 4, random_state: int = 42):
    """
    Aplica o dimensionamento (MinMaxScaler) e a clusterização (KMeans) replicando
    a lógica exata do script original. Retorna um dicionário unificado mapeando station_key -> cluster_id.
    """
    # --- Treino ---
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(df_train_mean)
    df_train_scaled_t = pd.DataFrame(X_train_scaled, index=df_train_mean.index, columns=df_train_mean.columns).T
    
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    df_train_scaled_t['Cluster'] = kmeans.fit_predict(df_train_scaled_t)
    
    # --- Teste ---
    # Mantendo fit_transform no teste para ser idêntico aos cálculos originais
    scaler_teste = MinMaxScaler()
    X_test_scaled = scaler_teste.fit_transform(df_test_mean)
    df_test_scaled_t = pd.DataFrame(X_test_scaled, index=df_test_mean.index, columns=df_test_mean.columns).T
    
    df_test_scaled_t['Cluster'] = kmeans.predict(df_test_scaled_t)
    
    return {**df_train_scaled_t['Cluster'].to_dict(), **df_test_scaled_t['Cluster'].to_dict()}

def add_clusters(df: pd.DataFrame, clusters_dict: dict, cluster_column_name: str):
    """
    Mapeia os clusters atribuídos às estações para o dataset principal, retornando os 
    dados combinados e uma lista de estações órfãs (caso existam).
    """
    df_com_clusters = df.copy()
    
    # Otimização com o `.map` em vez do iterrows do notebook original,
    # atingindo o mesmo resultado matematicamente.
    df_com_clusters[cluster_column_name] = df_com_clusters['station_key'].map(clusters_dict)
    lost_stations = df_com_clusters[df_com_clusters[cluster_column_name].isna()]['station_key'].unique().tolist()
    
    return df_com_clusters, lost_stations