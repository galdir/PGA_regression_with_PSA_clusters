"""
Script de execução para busca de hiperparâmetros (Hyperparameter Tuning).

Este módulo isola a etapa de otimização de modelos de Machine Learning
(Random Forest, XGBoost e Deep Neural Network) do pipeline principal.
Os resultados da busca são salvos automaticamente em arquivos JSON na pasta de resultados.

Como executar:
--------------
Via linha de comando, passe o modelo desejado e (opcionalmente) o número de tentativas:

    # Otimizar XGBoost com 100 tentativas
    python run_tuning.py --model xgb --trials 100

    # Otimizar Random Forest com 500 tentativas (padrão)
    python run_tuning.py --model rf
    
    # Otimizar Rede Neural (DNN) permitindo até 30 épocas de busca
    python run_tuning.py --model dnn --trials 30

    # Otimizar XGBoost usando uma feature específica de cluster
    python run_tuning.py --model xgb --cluster clusters_spectral_means

Argumentos:
-----------
--model  : (Obrigatório) 'rf' (Random Forest), 'xgb' (XGBoost) ou 'dnn' (Deep Neural Network).
--trials : (Opcional) Número máximo de iterações/épocas para a busca. Padrão: 500 para RF/XGB, 100 para DNN.
"""

import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config
from data_loader import load_pga_data, load_psa_data, get_common_records
from preprocessing import split_train_test_by_earthquake, create_features, remove_outliers
from clustering import split_psa_columns, calculate_psa_means, generate_clusters, add_clusters
from modeling import get_preprocessor, tune_random_forest, tune_xgboost, tune_dnn

def prepare_data(cluster_col=None):
    """
    Executa o pipeline de dados inicial necessário para obter os conjuntos 
    de treino limpos para a etapa de tuning.
    """
    print("Carregando e preparando os dados...")
    df_pga = load_pga_data(config.RAW_EARTHQUAKES_PGA_PATH)
    df_psa = load_psa_data(config.RAW_EARTHQUAKES_PSA_PATH)
    
    df_pga_common = get_common_records(df_pga, df_psa)
    df_pga_common = create_features(df_pga_common)
    
    df_train, df_test = split_train_test_by_earthquake(
        df_pga_common, 
        test_size=0.2, 
        random_state=config.RANDOM_STATE
    )
    
    df_train_clean = remove_outliers(
        df_train, 
        features=config.OUTLIER_FEATURES, 
        n_neighbors=200
    )
    
    if cluster_col and cluster_col != 'none':
        print(f"Processando e adicionando a feature de cluster: {cluster_col}...")
        train_ids = df_train['earthquake_id'].unique()
        test_ids = df_test['earthquake_id'].unique()
        
        df_psa_train, df_psa_test = split_psa_columns(df_psa, train_ids, test_ids)
        
        train_means = calculate_psa_means(df_psa_train)
        test_means = calculate_psa_means(df_psa_test)
        
        cluster_mapping = {
            'clusters_spectral_coefs': 'spectral_coefs',
            'clusters_spectral_means': 'spectral_means',
            'clusters_spectral_horiz_means': 'spectral_horiz_means',
            'clusters_spectral_vert_means': 'spectral_vert_means'
        }
        mean_key = cluster_mapping[cluster_col]
        k_val = config.OPTIMAL_K_VALUES.get(mean_key, 4)
        
        clusters_dict = generate_clusters(
            train_means[mean_key], test_means[mean_key], k=k_val, random_state=config.RANDOM_STATE
        )
        df_train_clean, lost_train = add_clusters(df_train_clean, clusters_dict, cluster_col)
        df_test, _ = add_clusters(df_test, clusters_dict, cluster_col)
        
        if lost_train:
            df_train_clean = df_train_clean.dropna(subset=[cluster_col])
            
    return df_train_clean, df_test

def main():
    parser = argparse.ArgumentParser(description="Executa a busca de hiperparâmetros por modelo.")
    parser.add_argument('--model', type=str, required=True, choices=['rf', 'xgb', 'dnn'],
                        help="Modelo para otimizar: 'rf' (Random Forest), 'xgb' (XGBoost) ou 'dnn' (Deep Neural Network)")
    parser.add_argument('--trials', type=int, default=None, 
                        help="Número de iterações/épocas máximas para a busca (Padrão: 300 para RF/XGB, 50 para DNN)")
    parser.add_argument('--cluster', type=str, default='none', choices=['none'] + config.CLUSTER_COLUMNS,
                        help="Atributo de cluster para otimizar (padrão: 'none' para modelo base)")
    args = parser.parse_args()

    # 1. Preparação dos Dados
    df_train, _ = prepare_data(cluster_col=args.cluster)
    
    X_train = df_train.drop(columns=['peak_ground_acceleration'])
    y_train = df_train['peak_ground_acceleration']
    groups = df_train['earthquake_id']

    if args.cluster != 'none':
        preprocessor = get_preprocessor(num_attributes=config.MODEL_NUM_FEATURES, cat_attributes=[args.cluster])
        output_suffix = f"_{args.model}_{args.cluster}"
    else:
        preprocessor = get_preprocessor(num_attributes=config.MODEL_NUM_FEATURES)
        output_suffix = f"_{args.model}"
    
    # 2. Configuração de Saída
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(config.RESULTS_DIR, f"tuning_results{output_suffix}.json")

    print(f"\nIniciando busca de hiperparâmetros para: {args.model.upper()} (Cluster: {args.cluster})")
    
    # 3. Execução da Busca por Modelo
    if args.model == 'rf':
        trials = args.trials if args.trials is not None else 500
        print(f"Máximo de iterações definidas: {trials}")
        study = tune_random_forest(X_train, y_train, groups, preprocessor, n_trials=trials, random_state=config.RANDOM_STATE)
        results = {
            "best_score_neg_rmse": study.best_value,
            "best_params": study.best_params
        }
        
    elif args.model == 'xgb':
        trials = args.trials if args.trials is not None else 500
        print(f"Máximo de iterações definidas: {trials}")
        study = tune_xgboost(X_train, y_train, groups, preprocessor, n_trials=trials, random_state=config.RANDOM_STATE)
        results = {
            "best_score_neg_rmse": study.best_value,
            "best_params": study.best_params
        }
        
    elif args.model == 'dnn':
        trials = args.trials if args.trials is not None else 100
        print(f"Máximo de épocas definidas: {trials}")
        # DNN requer divisão de validação extra a partir do treino, conforme notebook original
        # Utilizando GroupShuffleSplit para evitar data leakage de terremotos
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
        train_idx, valid_idx = next(gss.split(X_train, y_train, groups))
        
        X_train_dnn = X_train.iloc[train_idx]
        X_valid_dnn = X_train.iloc[valid_idx]
        y_train_dnn = y_train.iloc[train_idx]
        y_valid_dnn = y_train.iloc[valid_idx]
        
        tuner = tune_dnn(X_train_dnn, y_train_dnn, X_valid_dnn, y_valid_dnn, preprocessor, 
                         max_epochs=trials, project_name=f'keras_tuner{output_suffix}')
        
        best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
        results = {
            "best_score_val_rmse": best_trial.score,
            "best_params": best_trial.hyperparameters.values
        }
        
        print("\nSumário do melhor modelo Keras:")
        print(best_trial.summary())

    # 4. Salvando os resultados em disco
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\nBusca finalizada com sucesso!")
    print(f"Os melhores hiperparâmetros foram salvos em: {results_file}")

if __name__ == "__main__":
    main()