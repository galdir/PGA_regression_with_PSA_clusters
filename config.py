# Caminhos dos dados locais
RAW_EARTHQUAKES_PGA_PATH = 'earthquakes_pga.csv'
RAW_EARTHQUAKES_PSA_PATH = 'earthquakes_psa_earthquakeid.csv'

# Flags de controle de execução
LOAD_TRAINED_MODELS = False
HIPERPARAMETERS_TUNING = False

# Configurações gerais
RANDOM_STATE = 42

# Caminhos de diretórios
RESULTS_DIR = './paper_1_results_revised/'
TRAINED_PIPELINES_DIR = './paper_1_results_revised/trained_pipelines/'

# Features Numéricas para Modelagem (Exclui o Target, substitui ângulo bruto pelo log)
MODEL_NUM_FEATURES = [
    'magnitude', 'calculated_epicentral_distance', 'depth', 
    'log_incidence_angle', 'calculated_hypocentral_distance', 
    'source_off_shore'
]

# Features usadas na detecção de Outliers (Inclui o Target e as mesmas features do modelo)
OUTLIER_FEATURES = MODEL_NUM_FEATURES + ['peak_ground_acceleration']

CLUSTER_COLUMNS = [
    'clusters_spectral_coefs', 'clusters_spectral_means', 
    'clusters_spectral_horiz_means', 'clusters_spectral_vert_means'
]

# Valores otimizados de K (Método do Cotovelo) para cada tipo de espectro
OPTIMAL_K_VALUES = {
    'spectral_coefs': 5,
    'spectral_means': 4,
    'spectral_horiz_means': 4,
    'spectral_vert_means': 3
}