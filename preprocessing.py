import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria novas variáveis (Feature Engineering) como no código original."""
    df = df.copy()
    
    # Criação do logaritmo do ângulo de incidência
    if 'calculated_incidence_angle' in df.columns:
        df['log_incidence_angle'] = np.log(df['calculated_incidence_angle'])
    
    # Categorização de magnitude
    if 'magnitude' in df.columns:
        df["magnitude_cat"] = pd.cut(
            df["magnitude"],
            bins=[0., 6.08, 6.5, 7.2, np.inf],
            labels=[1, 2, 3, 4]
        )
        
    # Categorização de peak_ground_acceleration
    if 'peak_ground_acceleration' in df.columns:
        df["peak_ground_acceleration_cat"] = pd.cut(
            df['peak_ground_acceleration'],
            bins=[0., 2, 4, 6, 10, 18, 30, 80, np.inf],
            labels=[1, 2, 3, 4, 5, 6, 7, 8]
        )
        
    # Categorização de calculated_epicentral_distance
    if 'calculated_epicentral_distance' in df.columns:
        df["calculated_epicentral_distance_cat"] = pd.cut(
            df['calculated_epicentral_distance'],
            bins=[0., 130, 200, 280, 330, 430, np.inf],
            labels=[1, 2, 3, 4, 5, 6]
        )
    
    # Categorização de profundidade
    if 'depth' in df.columns:
        df["depth_cat"] = pd.cut(
            df['depth'],
            bins=[0., 10, 16, 30, 46, np.inf],
            labels=[1, 2, 3, 4, 5]
        )
        
    return df

def split_train_test_by_earthquake(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Separa os dados replicando exatamente a lógica cumulativa original.
    """
    eq_counts = df['earthquake_id'].value_counts().sample(frac=1, random_state=random_state)
    
    cumulative_records = eq_counts.cumsum()
    total_records = len(df)
    cumulative_percentage = cumulative_records / total_records
    
    train_ids = eq_counts[cumulative_percentage <= (1.0 - test_size)].index
    test_ids = eq_counts[cumulative_percentage > (1.0 - test_size)].index
    
    df_train = df[df['earthquake_id'].isin(train_ids)].copy().reset_index(drop=True)
    df_test = df[df['earthquake_id'].isin(test_ids)].copy().reset_index(drop=True)
    
    return df_train, df_test

def remove_outliers(df: pd.DataFrame, features: list, n_neighbors: int = 200, contamination: float = 0.001) -> pd.DataFrame:
    """
    Aplica o StandardScaler seguido do LocalOutlierFactor (LOF) 
    para remoção de outliers de acordo com o código original.
    """
    df_clean = df.copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])
    
    clf = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
    outlier_labels = clf.fit_predict(X_scaled)
    
    # lof retorna 1 para inliers e -1 para outliers
    inliers_mask = outlier_labels == 1
    
    return df_clean[inliers_mask].copy().reset_index(drop=True)