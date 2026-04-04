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
from modeling import get_preprocessor, tune_random_forest, tune_xgboost, tune_dnn

def prepare_data():
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
    
    return df_train_clean, df_test

def main():
    parser = argparse.ArgumentParser(description="Executa a busca de hiperparâmetros por modelo.")
    parser.add_argument('--model', type=str, required=True, choices=['rf', 'xgb', 'dnn'],
                        help="Modelo para otimizar: 'rf' (Random Forest), 'xgb' (XGBoost) ou 'dnn' (Deep Neural Network)")
    parser.add_argument('--trials', type=int, default=None, 
                        help="Número de iterações/épocas máximas para a busca (Padrão: 300 para RF/XGB, 50 para DNN)")
    args = parser.parse_args()

    # 1. Preparação dos Dados
    df_train, _ = prepare_data()
    
    X_train = df_train.drop(columns=['peak_ground_acceleration'])
    y_train = df_train['peak_ground_acceleration']
    groups = df_train['earthquake_id']

    preprocessor = get_preprocessor(num_attributes=config.MODEL_NUM_FEATURES)
    
    # 2. Configuração de Saída
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(config.RESULTS_DIR, f"tuning_results_{args.model}.json")

    print(f"\nIniciando busca de hiperparâmetros para: {args.model.upper()}")
    
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
        
        tuner = tune_dnn(X_train_dnn, y_train_dnn, X_valid_dnn, y_valid_dnn, preprocessor, max_epochs=trials)
        
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