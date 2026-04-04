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

    # Otimizar Random Forest com 50 tentativas (padrão)
    python run_tuning.py --model rf
    
    # Otimizar Rede Neural (DNN) permitindo até 30 épocas de busca
    python run_tuning.py --model dnn --trials 30

Argumentos:
-----------
--model  : (Obrigatório) 'rf' (Random Forest), 'xgb' (XGBoost) ou 'dnn' (Deep Neural Network).
--trials : (Opcional) Número máximo de iterações/épocas para a busca. O padrão é 50.
"""

import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

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
    df_pga = load_pga_data(config.RAW_EARTHQUAKES_PGA_URL)
    df_psa = load_psa_data(config.RAW_EARTHQUAKES_PSA_URL)
    
    df_pga_common = get_common_records(df_pga, df_psa)
    df_pga_common = create_features(df_pga_common)
    
    df_train, df_test = split_train_test_by_earthquake(
        df_pga_common, 
        test_size=0.2, 
        random_state=config.RANDOM_STATE
    )
    
    df_train_clean = remove_outliers(
        df_train, 
        features=config.SELECTED_ATTRIBUTES, 
        n_neighbors=200
    )
    
    return df_train_clean, df_test

def main():
    parser = argparse.ArgumentParser(description="Executa a busca de hiperparâmetros por modelo.")
    parser.add_argument('--model', type=str, required=True, choices=['rf', 'xgb', 'dnn'],
                        help="Modelo para otimizar: 'rf' (Random Forest), 'xgb' (XGBoost) ou 'dnn' (Deep Neural Network)")
    parser.add_argument('--trials', type=int, default=50, 
                        help="Número de iterações/épocas máximas para a busca (padrão: 50)")
    args = parser.parse_args()

    # 1. Preparação dos Dados
    df_train, _ = prepare_data()
    
    X_train = df_train.drop(columns=['peak_ground_acceleration'])
    y_train = df_train['peak_ground_acceleration']
    groups = df_train['earthquake_id']

    preprocessor = get_preprocessor(num_attributes=config.NUM_ATTRIBUTES)
    
    # 2. Configuração de Saída
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(config.RESULTS_DIR, f"tuning_results_{args.model}.json")

    print(f"\nIniciando busca de hiperparâmetros para: {args.model.upper()}")
    print(f"Máximo de iterações/épocas definidas: {args.trials}")
    
    # 3. Execução da Busca por Modelo
    if args.model == 'rf':
        study = tune_random_forest(X_train, y_train, preprocessor, n_trials=args.trials, random_state=config.RANDOM_STATE)
        results = {
            "best_score_neg_rmse": study.best_value,
            "best_params": study.best_params
        }
        
    elif args.model == 'xgb':
        study = tune_xgboost(X_train, y_train, groups, preprocessor, n_trials=args.trials, random_state=config.RANDOM_STATE)
        results = {
            "best_score_neg_rmse": study.best_value,
            "best_params": study.best_params
        }
        
    elif args.model == 'dnn':
        # DNN requer divisão de validação extra a partir do treino, conforme notebook original
        X_train_dnn, X_valid_dnn, y_train_dnn, y_valid_dnn = train_test_split(
            X_train, y_train, test_size=0.2, random_state=config.RANDOM_STATE
        )
        tuner = tune_dnn(X_train_dnn, y_train_dnn, X_valid_dnn, y_valid_dnn, preprocessor, max_epochs=args.trials)
        
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