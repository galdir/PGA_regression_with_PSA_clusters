import os
import urllib.request
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import SilhouetteVisualizer

import config
from main import set_global_seeds
from data_loader import load_pga_data, load_psa_data, get_common_records
from preprocessing import split_train_test_by_earthquake, create_features, remove_outliers
from clustering import split_psa_columns, calculate_psa_means, generate_clusters, add_clusters
from evaluation import load_pipelines

warnings.filterwarnings("ignore", category=FutureWarning)

# Configurações globais de plotagem
FIGURES_DIR = os.path.join(config.RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def download_shapefiles():
    """Baixa os shapefiles do México caso não existam localmente."""
    base_url = 'https://raw.githubusercontent.com/galdir/PGA_regression_with_PSA_clusters/main/'
    files = ['ne_110m_admin_0_countries.shp', 'ne_110m_admin_0_countries.shx']
    for f in files:
        if not os.path.exists(f):
            print(f"Baixando shapefile {f}...")
            urllib.request.urlretrieve(base_url + f, f)
    return 'ne_110m_admin_0_countries.shp'

def longitude_formatter(x, pos):
    return f'{x}°W'

def latitude_formatter(x, pos):
    return f'{x}°N'

def plot_eda_figures(df_pga_common, world):
    print("Gerando Figuras de Análise Exploratória (EDA)...")
    df = df_pga_common.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    locations = [[-99.1332, 19.4326, "Mexico City"]]

    # ---------------------------------------------------------
    # Figura 1: Mapa do Conjunto de Dados
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(6, 4))
    world.plot(ax=ax1, edgecolor='black', facecolor='green', linewidth=1, zorder=1, alpha=0.5)
    
    sns.scatterplot(ax=ax1, data=df, x='earthquake_longitude', y='earthquake_latitude',
                    size='magnitude', hue='magnitude', sizes=(40, 400), zorder=2,
                    palette=sns.color_palette("Reds", as_cmap=True), marker=(8,1,0))
    
    sns.scatterplot(ax=ax1, data=df, x='station_longitude', y='station_latitude',
                    zorder=3, color='black', s=30, label='Seismic Station')

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(longitude_formatter))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(latitude_formatter))
    
    ax1.set_xlabel('Longitude', labelpad=10)
    ax1.set_ylabel('Latitude', labelpad=10)
    ax1.set_xlim([-106, -90])
    ax1.set_ylim([13, 22])
    plt.xticks(rotation=45)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_01_mapa_conjunto_dados.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Figura 2: Histogramas de Atributos
    # ---------------------------------------------------------
    fig, axs = plt.subplots(4, 3, figsize=(9, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    features = [
        ('magnitude', 'Magnitude (Mw)'), ('peak_ground_acceleration', 'Peak ground acceleration (cm/s²)'), 
        ('depth', 'Depth (km)'), ('earthquake_latitude', 'Earthquake Latitude (°)'), 
        ('earthquake_longitude', 'Earthquake Longitude (°)'), ('station_latitude', 'Station Latitude (°)'), 
        ('station_longitude', 'Station Longitude (°)'), ('calculated_epicentral_distance', 'Epicentral Distance (km)'), 
        ('incidence_angle', 'Incidence Angle (°)'), ('calculated_azimuth', 'Azimuth (°)'), 
        ('date', 'Date'), ('calculated_hypocentral_distance', 'Hypocentral Distance (km)')
    ]
    
    for i, (col, label) in enumerate(features):
        row, col_idx = divmod(i, 3)
        if col == 'date':
            sns.histplot(data=df, x=col, kde=True, bins=8, ax=axs[row, col_idx])
            axs[row, col_idx].tick_params(axis='x', rotation=45)
        else:
            sns.histplot(data=df, x=col, kde=True, ax=axs[row, col_idx])
        axs[row, col_idx].set_xlabel(label)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_02_histogramas_atributos.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Figura 3: Estudo de Caso de Evento Específico
    # ---------------------------------------------------------
    one_event = df[df['date'].dt.strftime('%Y-%m-%d') == '2017-09-19']
    if not one_event.empty:
        fig, ax1 = plt.subplots(figsize=(5, 6))
        world.plot(ax=ax1, edgecolor='black', facecolor='green', linewidth=1, zorder=1, alpha=0.5)
        for lon, lat, name in locations:
            ax1.plot(lon, lat, 'ko', markersize=3)
            ax1.text(lon, lat, name, color='w', fontsize=8, ha='right', va='bottom')

        sns.scatterplot(ax=ax1, data=one_event, x='earthquake_longitude', y='earthquake_latitude',
                        color='red', s=1000, zorder=2, marker=(8,1,0))
        sns.scatterplot(ax=ax1, data=one_event, x='station_longitude', y='station_latitude',
                        zorder=3, hue='peak_ground_acceleration', size='peak_ground_acceleration',
                        palette=sns.color_palette("flare", as_cmap=True), sizes=(20, 200))
        
        ax1.set_xlim([-99, -97.5])
        ax1.set_ylim([18, 19.2])
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(longitude_formatter))
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(latitude_formatter))
        
        ax1.set_xlabel('Longitude', labelpad=10)
        ax1.set_ylabel('Latitude', labelpad=10)
        plt.legend(frameon=True, title='Peak Ground Acceleration', loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig_03_estudo_evento_2017.png'), dpi=300)
        plt.close()

def plot_psa_signals(df_psa, train_means):
    print("Gerando Figuras de Sinais PSA...")
    
    # ---------------------------------------------------------
    # Figura 4: Semelhança Intra-estação de PSAs
    # ---------------------------------------------------------
    cols = df_psa.columns
    cale_cols = [c for c in cols if c.startswith('CALE')]
    acad_cols = [c for c in cols if c.startswith('ACAD')]
    
    if cale_cols and acad_cols:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(df_psa['period(s)'].values, df_psa[cale_cols].values)
        axs[0].set_xlabel('Period (s)')
        axs[0].set_ylabel('Acceleration (cm/s²)')
        axs[0].set_title('Station: CALE')
        
        axs[1].plot(df_psa['period(s)'].values, df_psa[acad_cols].values)
        axs[1].set_xlabel('Period (s)')
        axs[1].set_ylabel('Acceleration (cm/s²)')
        axs[1].set_title('Station: ACAD')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig_04_semelhanca_psa.png'), dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Figura 5: Vetores de Pré-processamento
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    periods = df_psa['period(s)'].values[:len(train_means['spectral_coefs'])]
    
    axs[0,0].plot(periods, train_means['spectral_coefs'].values)
    axs[0,0].set_ylabel('Dimensionless')
    axs[0,0].set_title('(a) Spectral Ratio')
    
    axs[0,1].plot(periods, train_means['spectral_means'].values)
    axs[0,1].set_ylabel('cm/s²')
    axs[0,1].set_title('(b) Three Axis Spectra')
    
    axs[1,0].plot(periods, train_means['spectral_horiz_means'].values)
    axs[1,0].set_xlabel('Period (s)')
    axs[1,0].set_ylabel('cm/s²')
    axs[1,0].set_title('(c) Horizontal Spectra')
    
    axs[1,1].plot(periods, train_means['spectral_vert_means'].values)
    axs[1,1].set_xlabel('Period (s)')
    axs[1,1].set_ylabel('cm/s²')
    axs[1,1].set_title('(d) Vertical Spectra')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_05_vetores_pre_processamento.png'), dpi=300)
    plt.close()

def plot_clustering_results(train_means):
    print("Gerando Figuras de Clusterização...")
    
    # Prepara df de 'spectral_means' para o Silhouette
    df_means = train_means['spectral_means']
    scaler = MinMaxScaler()
    Xscaled = scaler.fit_transform(df_means)
    df_scaled_t = pd.DataFrame(Xscaled, index=df_means.index, columns=df_means.columns).T

    # ---------------------------------------------------------
    # Figura 6: Avaliação de Silhueta
    # ---------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for i, k in enumerate([2, 3, 4, 5]):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=config.RANDOM_STATE)
        q, mod = divmod(i, 2)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q][mod])
        visualizer.fit(df_scaled_t)
        ax[q][mod].set_title(f'K = {k}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_06_silhouette_spectral_means.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Figuras 7 a 10: Agrupamentos KMeans
    # ---------------------------------------------------------
    mapping = {
        'spectral_means': ('fig_07_kmeans_spectral_means.png', 4, 'Three Axis Spectra'),
        'spectral_coefs': ('fig_08_kmeans_spectral_coefs.png', 4, 'Spectral Ratio'),
        'spectral_horiz_means': ('fig_09_kmeans_spectral_horiz_means.png', 4, 'Horizontal Spectra'),
        'spectral_vert_means': ('fig_10_kmeans_spectral_vert_means.png', 4, 'Vertical Spectra')
    }
    
    for key, (filename, k_val, title) in mapping.items():
        df_m = train_means[key]
        sc = MinMaxScaler()
        X_s = sc.fit_transform(df_m)
        df_s_t = pd.DataFrame(X_s, index=df_m.index, columns=df_m.columns).T
        
        km = KMeans(n_clusters=k_val, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(df_s_t)
        X_vals = df_s_t.values
        
        fig, axs = plt.subplots(1, k_val, figsize=(12, 3), sharey=True)
        fig.suptitle(f'KMeans Clustering - {title}', y=1.05)
        
        for label in set(labels):
            cluster_signals = []
            for idx in range(len(labels)):
                if labels[idx] == label:
                    axs[label].plot(X_vals[idx], c="gray", alpha=0.4)
                    cluster_signals.append(X_vals[idx])
            if cluster_signals:
                axs[label].plot(np.average(np.vstack(cluster_signals), axis=0), c="red", linewidth=2)
            axs[label].set_title(f"Cluster {label}")
            
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()

def plot_regression_results(df_test, pipelines):
    print("Gerando Figuras de Resultados de Regressão...")
    
    def plot_scatter(model_name, filename, title):
        if model_name not in pipelines:
            print(f"Modelo '{model_name}' não encontrado nos pipelines salvos. Pulando {filename}.")
            return
            
        pipeline = pipelines[model_name]
        preds = pipeline.predict(df_test).flatten()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=np.log(df_test['peak_ground_acceleration']), y=np.log(np.abs(preds)), ax=ax, alpha=0.6)
        
        # Linha y=x de referência
        lims = [-4, 7]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        
        ax.set_xlabel('Measured ln(PGA)')
        ax.set_ylabel('Predicted ln(PGA)')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
        plt.close()

    # Figura 11
    plot_scatter('Random Forest', 'fig_11_scatter_baseline.png', 'Random Forest (Baseline)')
    
    # Figura 12
    plot_scatter('Random Forest with PSA clusters_spectral_horiz_means', 
                 'fig_12_scatter_com_cluster_horiz.png', 
                 'Random Forest (with Horizontal Spectra Cluster)')

def plot_regionalization_and_simulation(df_full, df_test, pipelines, world):
    print("Gerando Figuras de Regionalização e Simulação...")
    
    # ---------------------------------------------------------
    # Figura 13: Mapa de Regionalização por Cluster
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 6))
    world.plot(ax=ax1, edgecolor='black', facecolor='green', linewidth=1, zorder=1, alpha=0.5)
    
    sns.scatterplot(ax=ax1, data=df_full, x='station_longitude', y='station_latitude',
                    hue='clusters_spectral_horiz_means', s=50, zorder=2, 
                    palette=sns.color_palette('colorblind', n_colors=df_full['clusters_spectral_horiz_means'].nunique()))
    
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(longitude_formatter))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(latitude_formatter))
    
    ax1.set_xlabel('Longitude', labelpad=10)
    ax1.set_ylabel('Latitude', labelpad=10)
    ax1.set_xlim([-106, -90])
    ax1.set_ylim([13, 22])
    plt.xticks(rotation=45)
    plt.legend(frameon=True, title='Horizontal Spectra Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_13_mapa_regionalizacao.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Figura 14: Simulação de PGA por Cluster
    # ---------------------------------------------------------
    model_name = 'Random Forest with PSA clusters_spectral_horiz_means'
    if model_name in pipelines and not df_test.empty:
        pipeline = pipelines[model_name]
        
        # Selecionar um evento representativo do conjunto de teste (ex: primeiro registro)
        base_simulation = df_test.iloc[[0]].copy()
        
        simulations = []
        # Simular para cada cluster possível (0, 1, 2, 3)
        for i in range(4):
            sim_i = base_simulation.copy()
            sim_i['clusters_spectral_horiz_means'] = i
            pred = pipeline.predict(sim_i).flatten()[0]
            
            # Reverter a transformação logarítmica (TransformedTargetRegressor) para PGA normal (cm/s2)
            sim_i['pga_pred'] = np.exp(pred) 
            simulations.append(sim_i)
            
        df_sim = pd.concat(simulations).reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_sim, x="clusters_spectral_horiz_means", y="pga_pred", ax=ax, palette='viridis')
        ax.set_xlabel("Horizontal Spectra Cluster")
        ax.set_ylabel("Predicted PGA (cm/s²)")
        ax.set_title("PGA Simulation for a Hypothetical Earthquake by Cluster")
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig_14_simulacao_pga_cluster.png'), dpi=300)
        plt.close()
    else:
        print(f"Modelo {model_name} indisponível para gerar a simulação (Fig 14).")

def main():
    print("Iniciando a geração de figuras...")
    set_global_seeds(config.RANDOM_STATE)
    
    # 1. Obter shapefile do mapa
    shp_path = download_shapefiles()
    world = gpd.read_file(shp_path)
    
    # 2. Carregamento e Preparação de Dados (Determinístico)
    print("Recriando pipelines de dados...")
    df_pga = load_pga_data(config.RAW_EARTHQUAKES_PGA_PATH)
    df_psa = load_psa_data(config.RAW_EARTHQUAKES_PSA_PATH)
    
    df_pga_common = get_common_records(df_pga, df_psa)
    df_pga_common = create_features(df_pga_common)
    
    df_train, df_test = split_train_test_by_earthquake(df_pga_common, test_size=0.2, random_state=config.RANDOM_STATE)
    df_train_clean = remove_outliers(df_train, features=config.OUTLIER_FEATURES, n_neighbors=200)
    
    train_ids = df_train['earthquake_id'].unique()
    test_ids = df_test['earthquake_id'].unique()
    
    df_psa_train, df_psa_test = split_psa_columns(df_psa, train_ids, test_ids)
    train_means = calculate_psa_means(df_psa_train)
    test_means = calculate_psa_means(df_psa_test)
    
    # 3. Clusterização para aplicar no full dataset (para mapas de regionalização)
    cluster_mapping = {
        'spectral_coefs': 'clusters_spectral_coefs',
        'spectral_means': 'clusters_spectral_means',
        'spectral_horiz_means': 'clusters_spectral_horiz_means',
        'spectral_vert_means': 'clusters_spectral_vert_means'
    }
    
    df_full = df_pga_common.copy()
    
    for mean_key, cluster_col in cluster_mapping.items():
        k_val = config.OPTIMAL_K_VALUES.get(mean_key, 4)
        clusters_dict = generate_clusters(train_means[mean_key], test_means[mean_key], k=k_val, random_state=config.RANDOM_STATE)
        
        df_train_clean, _ = add_clusters(df_train_clean, clusters_dict, cluster_col)
        df_test, _ = add_clusters(df_test, clusters_dict, cluster_col)
        df_full, _ = add_clusters(df_full, clusters_dict, cluster_col)
        
    # Preencher orfãos temporariamente para os plots espaciais não quebrarem
    for col in cluster_mapping.values():
        df_full[col] = df_full[col].fillna(-1)
        
    # 4. Carregar pipelines de Machine Learning salvos
    pipelines = load_pipelines(config.TRAINED_PIPELINES_DIR)
    
    # 5. Gerar as Figuras Modulares
    plot_eda_figures(df_pga_common, world)
    plot_psa_signals(df_psa, train_means)
    plot_clustering_results(train_means)
    plot_regression_results(df_test, pipelines)
    plot_regionalization_and_simulation(df_full, df_test, pipelines, world)
    
    print(f"\nConcluído! Todas as figuras foram salvas em: {FIGURES_DIR}")

if __name__ == "__main__":
    main()