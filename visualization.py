import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd

# Funções utilitárias de formatação para mapas
def _longitude_formatter(x, p):
    return f'{x}°W'

def _latitude_formatter(x, p):
    return f'{x}°N'


def plot_psa_spectra(df_psa_merged: pd.DataFrame, output_path: str = None):
    """
    Plota as curvas espectrais gerais.
    """
    plt.figure(figsize=(4, 3))
    plt.plot(df_psa_merged['period(s)'], df_psa_merged.iloc[:, 1:-1])
    plt.xlabel('Period (s)')
    plt.ylabel('Acceleration (cm/s $^2$)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_regression_scatter(trained_pipelines: dict, test_set: pd.DataFrame, models: list, title: str, output_path: str = None):
    """
    Plota gráficos de dispersão (Medido vs Previsto) para múltiplos modelos.
    """
    fig, axes = plt.subplots(1, len(models), figsize=(12, 3), sharey=True)
    fig.suptitle(title)

    for i, model in enumerate(models):
        if model not in trained_pipelines:
            continue
            
        test_pred = trained_pipelines[model].predict(test_set)
        plot_title = model.split('with')[0] if 'with' in model else model

        if len(test_pred.shape) > 1:
            test_pred = test_pred[:, 0]
            
        sns.scatterplot(ax=axes[i], x=np.log(test_set['peak_ground_acceleration']), y=np.log(np.abs(test_pred)))

        # Linha de referência (y=x)
        lims = [-4, 7]
        axes[i].plot(lims, lims, 'r--', alpha=0.75, zorder=0)

        axes[i].set_xlabel('Measured ln(PGA)')
        if i == 0:
            axes[i].set_ylabel('Predicted ln(PGA)')
        axes[i].set_xlim(-4, 7)
        axes[i].set_ylim(-4, 7)
        axes[i].set_title(plot_title)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_error_analysis(test_set: pd.DataFrame, pred_without_cluster: np.ndarray, pred_with_cluster: np.ndarray, feature_x: str, feature_label: str, error_type: str = 'absolute', output_path: str = None):
    """
    Plota análise de erros (Absoluto ou Percentual) comparando o modelo com e sem clusters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    if error_type == 'percent':
        err_1 = np.abs((test_set['peak_ground_acceleration'] - pred_without_cluster) / test_set['peak_ground_acceleration'])
        err_2 = np.abs((test_set['peak_ground_acceleration'] - pred_with_cluster) / test_set['peak_ground_acceleration'])
        y_label = 'PGA error (%)'
    else:
        err_1 = np.abs(test_set['peak_ground_acceleration'] - pred_without_cluster)
        err_2 = np.abs(test_set['peak_ground_acceleration'] - pred_with_cluster)
        y_label = 'PGA error'

    sns.scatterplot(ax=axes[0], x=test_set[feature_x], y=err_1)
    axes[0].set_xlabel(feature_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_title('Without PSA Cluster')

    sns.scatterplot(ax=axes[1], x=test_set[feature_x], y=err_2)
    axes[1].set_xlabel(feature_label)
    axes[1].set_ylabel(y_label)
    axes[1].set_title('With Three Axis Spectra Cluster')

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_error_heatmap(test_set: pd.DataFrame, pred_without_cluster: np.ndarray, pred_with_cluster: np.ndarray, feature_x: str = 'magnitude', feature_label: str = 'Magnitude', output_path: str = None):
    """
    Plota um mapa de calor (histplot 2D) de erros percentuais.
    """
    percent_error_1 = (test_set['peak_ground_acceleration'] - pred_without_cluster) / test_set['peak_ground_acceleration']
    percent_error_2 = (test_set['peak_ground_acceleration'] - pred_with_cluster) / test_set['peak_ground_acceleration']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    sns.histplot(ax=axes[0], x=test_set[feature_x], y=np.abs(percent_error_1), bins=30, cbar=True, cmap='viridis')
    axes[0].set_ylabel('PGA error (%)')
    axes[0].set_title('Without PSA Cluster')
    axes[0].set_xlabel(feature_label)

    sns.histplot(ax=axes[1], x=test_set[feature_x], y=np.abs(percent_error_2), bins=30, cbar=True, cmap='viridis')
    axes[1].set_xlabel(feature_label)
    axes[1].set_title('With Three Axis Spectra Cluster')

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_cluster_map(df_com_clusters: pd.DataFrame, world_map_gdf, locations: list = None, cluster_col: str = 'clusters_spectral_means', zoom: bool = False, output_path: str = None):
    """
    Plota o mapa com a localização das estações coloridas pelo cluster correspondente.
    Usa um GeoDataFrame do geopandas/cartopy (world_map_gdf).
    """
    fig, ax1 = plt.subplots(figsize=(7 if not zoom else 6, 4))

    # Plote o contorno do mapa base (Geopandas object expected here)
    world_map_gdf.plot(ax=ax1, edgecolor='black', facecolor='green', linewidth=1, zorder=1, alpha=0.5)

    if locations and zoom:
        for lon, lat, name in locations:
            ax1.plot(lon, lat, 'ko', markersize=3)
            ax1.text(lon, lat, name, color='w', fontsize=8, ha='right', va='bottom')

    # Scatterplot das estações
    sns.scatterplot(
        ax=ax1,
        data=df_com_clusters,
        x='station_longitude',
        y='station_latitude',
        hue=cluster_col,
        s=50,
        zorder=2,
    )

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))  
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))  
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(_longitude_formatter))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(_latitude_formatter))

    ax1.set_xlabel('Longitude', labelpad=10)
    ax1.set_ylabel('Latitude', labelpad=10)
    
    if zoom:
        plt.xlim([-99, -97.5])
        plt.ylim([18, 19.2])
    else:
        plt.xticks(rotation=45)
        plt.xlim([-106, -90])
        plt.ylim([13, 22])

    plt.legend(frameon=True, title='Station Cluster' if not zoom else 'Cluster')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
