# URLs dos dados
RAW_EARTHQUAKES_PGA_URL = 'https://raw.githubusercontent.com/galdir/PGA_regression_with_PSA_clusters/main/earthquakes_pga.csv'
RAW_EARTHQUAKES_PSA_URL = 'https://raw.githubusercontent.com/galdir/PGA_regression_with_PSA_clusters/main/earthquakes_psa_earthquakeid.csv'

# Flags de controle de execução
LOAD_TRAINED_MODELS = False
HIPERPARAMETERS_TUNING = False

# Configurações gerais
RANDOM_STATE = 42

# Caminhos de diretórios
RESULTS_DIR = './paper_1_results_revised/'
TRAINED_PIPELINES_DIR = './paper_1_results_revised/trained_pipelines/'

# Definição de Features
SELECTED_ATTRIBUTES = [
    'magnitude', 'calculated_epicentral_distance', 'depth', 
    'calculated_incidence_angle', 'calculated_hypocentral_distance', 
    'peak_ground_acceleration', 'source_off_shore', 'calculated_azimuth'
]

NUM_ATTRIBUTES = [
    'magnitude', 'calculated_epicentral_distance', 'depth', 
    'calculated_incidence_angle', 'calculated_hypocentral_distance', 
    'source_off_shore', 'log_incidence_angle'
]

CLUSTER_COLUMNS = [
    'clusters_spectral_coefs', 'clusters_spectral_means', 
    'clusters_spectral_horiz_means', 'clusters_spectral_vert_means'
]