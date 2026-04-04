"""
Script para análise e seleção de clusters (K).

Este script carrega as médias espectrais dos sinais de treino, 
avalia agrupamentos (KMeans) variando de K=2 a K=8 e plota gráficos 
das métricas Elbow (Inércia) e Silhouette Score.

Os gráficos gerados são salvos no diretório configurado, 
substituindo a necessidade da análise visual do Jupyter Notebook.
"""

import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer

import config
from data_loader import load_pga_data, load_psa_data, get_common_records
from preprocessing import split_train_test_by_earthquake, create_features, remove_outliers
from clustering import split_psa_columns, calculate_psa_means

def plot_cluster_metrics(df_train_mean: pd.DataFrame, title: str, output_path: str) -> tuple[int, int]:
    """Gera e salva o gráfico combinando o Método do Cotovelo e Silhouette Score."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(df_train_mean)
    df_scaled_t = pd.DataFrame(X_train_scaled, index=df_train_mean.index, columns=df_train_mean.columns).T
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, 9)
    
    best_k = -1
    best_score = -1
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(df_scaled_t)
        inertias.append(kmeans.inertia_)
        
        score = silhouette_score(df_scaled_t, labels)
        silhouette_scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
            
    # Usando KElbowVisualizer num eixo isolado apenas para extrair matematicamente o K do cotovelo
    fig_dummy, ax_dummy = plt.subplots()
    km_elbow = KMeans(random_state=config.RANDOM_STATE, n_init=10)
    elbow_vis = KElbowVisualizer(km_elbow, k=(2, 9), timings=False, ax=ax_dummy)
    elbow_vis.fit(df_scaled_t)
    elbow_k = elbow_vis.elbow_value_
    plt.close(fig_dummy)
        
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Elbow)', color=color)
    ax1.plot(k_range, inertias, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette_scores, marker='s', linestyle='--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Cluster Evaluation: {title}')
    fig.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico salvo em: {output_path}")
    
    return best_k, elbow_k

def plot_silhouette_visualizers(df_train_mean: pd.DataFrame, title: str, output_path: str):
    """Gera o grid 2x2 de SilhouetteVisualizer replicando a lógica do notebook original."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(df_train_mean)
    df_scaled_t = pd.DataFrame(X_train_scaled, index=df_train_mean.index, columns=df_train_mean.columns).T
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f'Silhouette Analysis: {title}')
    
    axes_flat = axes.flatten()
    
    for idx, i in enumerate([2, 3, 4, 5]):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=config.RANDOM_STATE)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=axes_flat[idx])
        visualizer.fit(df_scaled_t)
        visualizer.finalize()
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico SilhouetteVisualizer salvo em: {output_path}")

def main():
    print("Carregando e preparando os dados de PSA...")
    df_pga = load_pga_data(config.RAW_EARTHQUAKES_PGA_PATH)
    df_psa = load_psa_data(config.RAW_EARTHQUAKES_PSA_PATH)
    
    df_pga_common = get_common_records(df_pga, df_psa)
    df_pga_common = create_features(df_pga_common)
    
    df_train, df_test = split_train_test_by_earthquake(df_pga_common, test_size=0.2, random_state=config.RANDOM_STATE)
    df_train_clean = remove_outliers(df_train, features=config.OUTLIER_FEATURES, n_neighbors=200)
    
    train_ids = df_train_clean['earthquake_id'].unique()
    test_ids = df_test['earthquake_id'].unique()
    
    df_psa_train, _ = split_psa_columns(df_psa, train_ids, test_ids)
    train_means = calculate_psa_means(df_psa_train)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("Analisando as métricas para os diferentes agrupamentos espectrais...")
    
    analyses = {
        'spectral_coefs': 'Spectral Coefficients',
        'spectral_means': 'Three Axis Spectra Means',
        'spectral_horiz_means': 'Horizontal Spectra Means',
        'spectral_vert_means': 'Vertical Spectra Means'
    }
    
    optimal_sil_ks = {}
    optimal_elbow_ks = {}
    
    for mean_key, title in analyses.items():
        # 1. Gráficos combinados e cálculo das métricas
        best_sil_k, elbow_k = plot_cluster_metrics(train_means[mean_key], title, os.path.join(config.RESULTS_DIR, f'k_evaluation_{mean_key}.png'))
        optimal_sil_ks[title] = best_sil_k
        optimal_elbow_ks[title] = elbow_k
        
        # 2. Gráficos de Silhouette Visualizer (K=2 a K=5)
        plot_silhouette_visualizers(train_means[mean_key], title, os.path.join(config.RESULTS_DIR, f'silhouette_visualizer_{mean_key}.png'))
        
    print("\n" + "="*55)
    print("RESUMO DOS MELHORES Ks (Silhouette vs Elbow)")
    print("="*55)
    for title in analyses.values():
        print(f" -> {title}:")
        print(f"      Maior Silhouette Score : K = {optimal_sil_ks[title]}")
        print(f"      Método do Cotovelo     : K = {optimal_elbow_ks[title]}")
        
    most_common_sil = Counter(optimal_sil_ks.values()).most_common(1)[0][0]
    most_common_elbow = Counter(optimal_elbow_ks.values()).most_common(1)[0][0]
    
    print(f"\n💡 Sugestão de K ÚNICO (Moda Silhouette): K = {most_common_sil}")
    print(f"💡 Sugestão de K ÚNICO (Moda Cotovelo)  : K = {most_common_elbow}\n")
        
if __name__ == "__main__":
    main()