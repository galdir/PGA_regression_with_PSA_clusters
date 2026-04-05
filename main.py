"""
Módulo principal de execução do pipeline de Machine Learning para predição de PGA.

Este projeto testa a hipótese de que a utilização de dados de clusterização 
baseados na forma dos sinais de Aceleração Pseudo-Espectral (PSA) das estações 
sísmicas ajuda a reduzir o erro de predição da Aceleração de Pico no Solo 
(PGA - Peak Ground Acceleration).

O fluxo orquestrado por este script inclui:
1. Carregamento e cruzamento dos dados brutos (PGA e PSA).
2. Engenharia de atributos (Feature Engineering).
3. Separação de Treino/Teste agrupada por terremoto (para evitar vazamento de dados).
4. Remoção de outliers exclusivamente no conjunto de treino.
5. Processamento de espectros PSA e geração de features de clusterização (KMeans).
6. Treinamento e avaliação de vários modelos (Regressão Linear, Random Forest, XGBoost, DNN),
   comparando o desempenho dos modelos "Base" (sem PSA) contra os modelos "Com Cluster".
7. Salvamento dos modelos treinados, métricas de avaliação e visualizações comparativas.

Como executar:
    python main.py
"""

import config
from data_loader import load_pga_data, load_psa_data, get_common_records, get_psa_station_earthquake_ids
from preprocessing import split_train_test_by_earthquake, create_features, remove_outliers
from clustering import split_psa_columns, calculate_psa_means, generate_clusters, add_clusters

def main():
    print("1. Carregando os dados...")
    df_pga = load_pga_data(config.RAW_EARTHQUAKES_PGA_PATH)
    df_psa = load_psa_data(config.RAW_EARTHQUAKES_PSA_PATH)
    psa_ids = get_psa_station_earthquake_ids(df_psa)
    print(f"   -> Total de registros PGA originais: {len(df_pga)}")
    print(f"   -> Total de registros PSA originais (estações-terremoto únicas): {len(psa_ids)}")
    print(f"   -> Terremotos únicos na base PGA: {df_pga['earthquake_id'].nunique()}")
    
    # Verificação de registros ausentes como no notebook original
    pga_ids = set(df_pga['station_earthquakeid'])
    missing_in_pga = psa_ids - pga_ids
    missing_in_psa = pga_ids - psa_ids
    print(f"   -> Registros em PSA mas ausentes em PGA: {len(missing_in_pga)}")
    print(f"   -> Registros em PGA mas ausentes em PSA: {len(missing_in_psa)}")
    
    print("\n2. Filtrando registros comuns entre PGA e PSA...")
    df_pga_common = get_common_records(df_pga, df_psa)
    print(f"   -> Total de registros PGA após filtro: {len(df_pga_common)}")
    
    print("\n3. Aplicando Engenharia de Atributos (Feature Engineering)...")
    df_pga_common = create_features(df_pga_common)
    print("   -> Novas features criadas com sucesso para o conjunto completo.")
    
    print("\n4. Realizando o split (Treino/Teste) agrupado por earthquake_id...")
    df_train, df_test = split_train_test_by_earthquake(
        df_pga_common, 
        test_size=0.2, 
        random_state=config.RANDOM_STATE
    )
    total_records = len(df_pga_common)
    print(f"   -> Total records: {total_records}")
    print(f"   -> Train records: {len(df_train)} ({len(df_train)/total_records:.1%})")
    print(f"   -> Test records: {len(df_test)} ({len(df_test)/total_records:.1%})")
    
    # Verificações de vazamento de dados (Data Leakage)
    train_eqs = df_train['earthquake_id'].nunique()
    test_eqs = df_test['earthquake_id'].nunique()
    overlap = set(df_train['earthquake_id']).intersection(set(df_test['earthquake_id']))
    print(f"   -> Terremotos únicos - Treino: {train_eqs} | Teste: {test_eqs}")
    print(f"   -> Sobreposição de terremotos entre Treino e Teste (deve ser 0): {len(overlap)}")
    
    # Estatísticas das novas features
    print("   -> Distribuição das novas features (Conjunto de Treino):")
    if 'magnitude_cat' in df_train.columns:
        print(f"      - Categorias de Magnitude:\n{df_train['magnitude_cat'].value_counts().to_string()}")
    if 'depth_cat' in df_train.columns:
        print(f"      - Categorias de Profundidade:\n{df_train['depth_cat'].value_counts().to_string()}")
    
    print("\n5. Removendo Outliers (Apenas no conjunto de Treino)...")
    df_train_clean = remove_outliers(
        df_train, 
        features=config.OUTLIER_FEATURES, 
        n_neighbors=200
    )
    print(f"   -> Tamanho do Treino após remoção de outliers: {len(df_train_clean)}")
    outliers_removidos = len(df_train) - len(df_train_clean)
    perc_outliers = (outliers_removidos / len(df_train)) * 100
    print(f"   -> Foram removidos {outliers_removidos} outliers do conjunto de treino ({perc_outliers:.2f}%).")
    
    print("\n6. Processando e Clusterizando os dados de PSA...")
    train_ids = df_train['earthquake_id'].unique()
    test_ids = df_test['earthquake_id'].unique()
    
    print("   -> Separando colunas PSA entre treino e teste...")
    df_psa_train, df_psa_test = split_psa_columns(df_psa, train_ids, test_ids)
    print(f"      - Total colunas originais PSA: {len(df_psa.columns)}")
    print(f"      - Colunas PSA destinadas ao Treino: {len(df_psa_train.columns)}")
    print(f"      - Colunas PSA destinadas ao Teste: {len(df_psa_test.columns)}")
    
    print("   -> Calculando médias espectrais (V, EW, NS)...")
    train_means = calculate_psa_means(df_psa_train)
    test_means = calculate_psa_means(df_psa_test)
    
    # Mapeamento das chaves das médias para os nomes das colunas definidos no config
    cluster_mapping = {
        'spectral_coefs': 'clusters_spectral_coefs',
        'spectral_means': 'clusters_spectral_means',
        'spectral_horiz_means': 'clusters_spectral_horiz_means',
        'spectral_vert_means': 'clusters_spectral_vert_means'
    }
    
    print("   -> Gerando e atribuindo clusters (KMeans k=4)...")
    for mean_key, cluster_col in cluster_mapping.items():
        print(f"      - Processando '{cluster_col}'...")
        clusters_dict = generate_clusters(
            train_means[mean_key], 
            test_means[mean_key], 
            k=4, 
            random_state=config.RANDOM_STATE
        )
        df_train_clean, lost_train = add_clusters(df_train_clean, clusters_dict, cluster_col)
        df_test, lost_test = add_clusters(df_test, clusters_dict, cluster_col)
        if lost_train or lost_test:
            print(f"        * Aviso: {len(lost_train)} estações sem cluster no treino e {len(lost_test)} no teste.")
        
    print("   -> Clusterização concluída. Datasets de treino e teste agora possuem as features de cluster.")

    print("\n7. Preparando para Modelagem...")
    from modeling import (
        get_preprocessor, build_linear_regression, build_polynomial_regression, 
        build_elasticnet, build_random_forest, build_xgboost, build_dnn_model
    )
    from evaluation import model_experiment, save_pipelines, save_experiments_results
    import os
    import json
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import root_mean_squared_error, r2_score
    from scipy import stats
    import numpy as np
    from sklearn.pipeline import Pipeline
    
    def get_tuned_params(model_key):
        """Lê os parâmetros do arquivo gerado pelo run_tuning.py, caso exista."""
        param_file = os.path.join(config.RESULTS_DIR, f"tuning_results_{model_key}.json")
        if os.path.exists(param_file):
            with open(param_file, 'r', encoding='utf-8') as f:
                return json.load(f).get('best_params', {})
        return {}
    
    experiments = {}
    trained_pipelines = {}
    
    def train_evaluate_dnn(df_train, df_test, preprocessor, model_name, dnn_params):
        print(f"Executando experimento: {model_name}...")
        
        X_train_dnn, X_valid_dnn, y_train_dnn, y_valid_dnn = train_test_split(
            df_train, df_train['peak_ground_acceleration'], test_size=0.2, random_state=config.RANDOM_STATE
        )
        
        tf_x_train = preprocessor.fit_transform(X_train_dnn)
        tf_x_valid = preprocessor.transform(X_valid_dnn)
        
        dnn_kwargs = {}
        if 'n_hidden' in dnn_params: dnn_kwargs['n_hidden_layers'] = dnn_params['n_hidden']
        if 'n_neurons' in dnn_params: dnn_kwargs['n_neurons'] = dnn_params['n_neurons']
        if 'activation' in dnn_params: dnn_kwargs['activation'] = dnn_params['activation']
        if 'learning_rate' in dnn_params: dnn_kwargs['learning_rate'] = dnn_params['learning_rate']
        if 'dropout_rate' in dnn_params: dnn_kwargs['dropout_rate'] = dnn_params['dropout_rate']
        
        model = build_dnn_model(**dnn_kwargs)
        es = EarlyStopping(monitor='val_rmse', mode='min', verbose=0, patience=10, restore_best_weights=True)
        
        y_train_dnn_log = np.log(y_train_dnn)
        y_valid_dnn_log = np.log(y_valid_dnn)
        
        model.fit(tf_x_train, y_train_dnn_log, epochs=100, callbacks=[es], validation_data=(tf_x_valid, y_valid_dnn_log), verbose=0)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        train_predictions = np.exp(pipeline.predict(df_train)).flatten()
        train_rmse = root_mean_squared_error(df_train['peak_ground_acceleration'], train_predictions)
        print(f"  -> Train RMSE: {train_rmse:.4f}")
        
        predictions = np.exp(pipeline.predict(df_test)).flatten()
        test_rmse = root_mean_squared_error(df_test['peak_ground_acceleration'], predictions)
        print(f"  -> Test RMSE: {test_rmse:.4f}")
        
        r2 = r2_score(df_test['peak_ground_acceleration'], predictions)
        print(f"  -> Test R2: {r2:.4f}")
        
        confidence = 0.95
        squared_errors = (predictions.flatten() - df_test['peak_ground_acceleration'].values) ** 2
        conf_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                 loc=np.mean(squared_errors),
                                 scale=stats.sem(squared_errors)))
        
        experiments[model_name] = [test_rmse, r2, conf_interval[0], conf_interval[1]]
        trained_pipelines[model_name] = pipeline

    rf_params = get_tuned_params('rf')
    xgb_params = get_tuned_params('xgb')
    dnn_params = get_tuned_params('dnn')

    # Modelos Base (Sem features de Cluster)
    print("   -> Treinando modelos BASE (Sem features de Cluster)")
    preprocessor_base = get_preprocessor(num_attributes=config.MODEL_NUM_FEATURES)
    
    # Linear Regression
    lr_base = build_linear_regression(preprocessor_base)
    model_experiment(df_train_clean, df_test, lr_base, experiments, 'Linear Regression', target_col='peak_ground_acceleration')
    trained_pipelines['Linear Regression'] = lr_base

    # Polynomial Regression
    poly_base = build_polynomial_regression(preprocessor_base)
    model_experiment(df_train_clean, df_test, poly_base, experiments, 'Polynomial Regression', target_col='peak_ground_acceleration')
    trained_pipelines['Polynomial Regression'] = poly_base

    # Polynomial ElasticNet
    elastic_base = build_elasticnet(preprocessor_base)
    model_experiment(df_train_clean, df_test, elastic_base, experiments, 'Polynomial ElasticNet', target_col='peak_ground_acceleration')
    trained_pipelines['Polynomial ElasticNet'] = elastic_base

    # Random Forest
    rf_base = build_random_forest(preprocessor_base, **rf_params)
    model_experiment(df_train_clean, df_test, rf_base, experiments, 'Random Forest', target_col='peak_ground_acceleration')
    trained_pipelines['Random Forest'] = rf_base

    # XGBoost
    xgb_base = build_xgboost(preprocessor_base, **xgb_params)
    model_experiment(df_train_clean, df_test, xgb_base, experiments, 'XGBoost', target_col='peak_ground_acceleration')
    trained_pipelines['XGBoost'] = xgb_base
    
    # Deep Neural Network
    train_evaluate_dnn(df_train_clean, df_test, preprocessor_base, 'Deep Neural Network', dnn_params)
    
    # Modelos com features de Cluster
    print("\n   -> Treinando modelos COM features de Cluster")
    for cluster_col in config.CLUSTER_COLUMNS:
        if cluster_col not in df_train_clean.columns:
            continue
            
        print(f"      - Cluster: {cluster_col}")
        preprocessor_cluster = get_preprocessor(num_attributes=config.MODEL_NUM_FEATURES, cat_attributes=[cluster_col])
        
        lr_cluster = build_linear_regression(preprocessor_cluster)
        model_experiment(df_train_clean, df_test, lr_cluster, experiments, f'Linear Regression with PSA {cluster_col}', target_col='peak_ground_acceleration')
        trained_pipelines[f'Linear Regression with PSA {cluster_col}'] = lr_cluster

        poly_cluster = build_polynomial_regression(preprocessor_cluster)
        model_experiment(df_train_clean, df_test, poly_cluster, experiments, f'Polynomial Regression with PSA {cluster_col}', target_col='peak_ground_acceleration')
        trained_pipelines[f'Polynomial Regression with PSA {cluster_col}'] = poly_cluster

        elastic_cluster = build_elasticnet(preprocessor_cluster)
        model_experiment(df_train_clean, df_test, elastic_cluster, experiments, f'Polynomial ElasticNet with PSA {cluster_col}', target_col='peak_ground_acceleration')
        trained_pipelines[f'Polynomial ElasticNet with PSA {cluster_col}'] = elastic_cluster

        rf_cluster = build_random_forest(preprocessor_cluster, **rf_params)
        model_experiment(df_train_clean, df_test, rf_cluster, experiments, f'Random Forest with PSA {cluster_col}', target_col='peak_ground_acceleration')
        trained_pipelines[f'Random Forest with PSA {cluster_col}'] = rf_cluster
        
        xgb_cluster = build_xgboost(preprocessor_cluster, **xgb_params)
        model_experiment(df_train_clean, df_test, xgb_cluster, experiments, f'XGBoost with PSA {cluster_col}', target_col='peak_ground_acceleration')
        trained_pipelines[f'XGBoost with PSA {cluster_col}'] = xgb_cluster
        
        # Deep Neural Network
        train_evaluate_dnn(df_train_clean, df_test, preprocessor_cluster, f'Deep Neural Network with PSA {cluster_col}', dnn_params)

    print("\n8. Salvando Resultados e Pipelines...")
    save_experiments_results(experiments, os.path.join(config.RESULTS_DIR, 'experiments_results.csv'))
    save_pipelines(trained_pipelines, config.TRAINED_PIPELINES_DIR)
    
    print("\n--- Tabela Final de Resultados (Ordenada por Test RMSE) ---")
    import pandas as pd
    df_results = pd.DataFrame.from_dict(
        experiments, 
        orient='index', 
        columns=['Test RMSE', 'R2', 'CI Lower', 'CI Upper']
    )
    df_results.index.name = 'Model'
    df_results.sort_values(by='Test RMSE', ascending=True, inplace=True)
    print(df_results.to_string())

    print("\n9. Visualizações Finais...")
    from visualization import plot_regression_scatter, plot_error_analysis
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Gráficos de Scatter
    scatter_models = ['Random Forest', f'Random Forest with PSA {config.CLUSTER_COLUMNS[0]}']
    if all(m in trained_pipelines for m in scatter_models):
        print(f"   -> Gerando Scatter Plot para: {scatter_models}")
        plot_regression_scatter(trained_pipelines, df_test, scatter_models, "Scatter RF - Base vs Cluster", 
                                output_path=os.path.join(config.RESULTS_DIR, "scatter_rf.png"))
                            
    # Análise de Erros (XGBoost)
    xgb_base_name = 'XGBoost'
    xgb_cluster_name = f'XGBoost with PSA {config.CLUSTER_COLUMNS[0]}'
    if xgb_base_name in trained_pipelines and xgb_cluster_name in trained_pipelines:
        print("   -> Gerando Gráficos de Análise de Erro (XGBoost)")
        pred_base = trained_pipelines[xgb_base_name].predict(df_test)
        pred_cluster = trained_pipelines[xgb_cluster_name].predict(df_test)
        
        plot_error_analysis(df_test, pred_base, pred_cluster, 'magnitude', 'Magnitude', 
                            output_path=os.path.join(config.RESULTS_DIR, "error_analysis_xgb_mag.png"))
        plot_error_analysis(df_test, pred_base, pred_cluster, 'calculated_epicentral_distance', 'Distância Epicentral (km)', 
                            output_path=os.path.join(config.RESULTS_DIR, "error_analysis_xgb_dist.png"))

    print("\n--- Pipeline Completo Finalizado com Sucesso! ---")

if __name__ == "__main__":
    main()