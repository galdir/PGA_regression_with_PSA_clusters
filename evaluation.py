import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib
import os
import copy
from tensorflow.keras.models import load_model

def model_experiment(train_set: pd.DataFrame, test_set: pd.DataFrame, pipeline, experiments_dict: dict, experiment_name: str, target_col: str = 'peak_ground_acceleration', confidence: float = 0.95):
    """
    Treina o pipeline, calcula as predições e métricas (RMSE, R2, Intervalo de Confiança),
    e armazena os resultados em um dicionário.
    """
    print(f"Executando experimento: {experiment_name}...")
    
    # Treinamento
    pipeline.fit(train_set, train_set[target_col])
    
    # Avaliação no treino
    train_predictions = pipeline.predict(train_set)
    train_rmse = root_mean_squared_error(train_set[target_col], train_predictions)
    print(f"  -> Train RMSE: {train_rmse:.4f}")

    # Avaliação no teste
    predictions = pipeline.predict(test_set)
    test_rmse = root_mean_squared_error(test_set[target_col], predictions)
    print(f"  -> Test RMSE: {test_rmse:.4f}")
    
    r2 = r2_score(test_set[target_col], predictions)
    print(f"  -> Test R2: {r2:.4f}")

    # Cálculo do intervalo de confiança para os erros ao quadrado
    y_true = test_set[target_col].values.flatten()
    y_pred = np.array(predictions).flatten()
    squared_errors = (y_pred - y_true) ** 2
        
    conf_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc=np.mean(squared_errors),
                             scale=stats.sem(squared_errors)))
    
    # Armazenar resultados
    experiments_dict[experiment_name] = [test_rmse, r2, conf_interval[0], conf_interval[1]]

    return test_rmse, r2, pipeline, experiments_dict


def save_pipelines(trained_pipelines: dict, trained_pipelines_path: str):
    """
    Salva os pipelines treinados no disco. 
    Lida especificamente com a separação do modelo Keras (DNN) do joblib.
    """
    if not os.path.exists(trained_pipelines_path):
        os.makedirs(trained_pipelines_path)

    trained_pipelines_copy = copy.deepcopy(trained_pipelines)

    for pipeline_name, pipeline in trained_pipelines_copy.items():
        print(f"Salvando pipeline: {pipeline_name}")
        filename = os.path.join(trained_pipelines_path, f'{pipeline_name}.joblib')
        
        if pipeline_name.startswith('Deep Neural Network'):
            # O modelo DNN não é bem suportado pelo joblib diretamente no pipeline Scikit-Learn.
            model = pipeline.named_steps['model']
            keras_path = os.path.join(trained_pipelines_path, f'{pipeline_name}.keras')
            model.save(keras_path)
            
            # Remover o modelo para salvar o resto do pipeline
            pipeline.steps[-1] = ('model', None)
            joblib.dump(pipeline, filename)
            
            # Restaurar o modelo
            pipeline.steps[-1] = ('model', model)
        else:
            joblib.dump(pipeline, filename)


def load_pipelines(trained_pipelines_path: str) -> dict:
    """
    Carrega pipelines previamente salvos.
    Reconstroi pipelines de Deep Neural Network associando o arquivo .keras ao .joblib.
    """
    trained_pipelines = {}
    
    if not os.path.exists(trained_pipelines_path):
        print(f"Caminho não encontrado: {trained_pipelines_path}")
        return trained_pipelines

    for file_path in os.listdir(trained_pipelines_path):
        full_path = os.path.join(trained_pipelines_path, file_path)
        
        if os.path.isfile(full_path) and file_path.endswith('.joblib'):
            pipeline_name = file_path.split('.joblib')[0]
            print(f"Carregando pipeline: {pipeline_name}")
            
            pipeline = joblib.load(full_path)
            
            if pipeline_name.startswith('Deep Neural Network'):
                keras_path = os.path.join(trained_pipelines_path, f'{pipeline_name}.keras')
                dnn_model = load_model(keras_path, compile=False)
                # Adiciona o modelo de volta ao último passo
                pipeline.steps[-1] = ('model', dnn_model)
                
            trained_pipelines[pipeline_name] = pipeline

    return trained_pipelines


def save_experiments_results(experiments_dict: dict, filename: str):
    """Salva os resultados dos experimentos em um arquivo CSV para facilitar a leitura."""
    df = pd.DataFrame.from_dict(experiments_dict, orient='index', 
                                columns=['test_rmse', 'r2', 'conf_interval_lower', 'conf_interval_upper'])
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df.to_csv(filename, index_label='model_name')
    print(f"Resultados dos experimentos salvos em: {filename}")


def load_experiments_results(filename: str) -> dict:
    """Carrega os resultados dos experimentos de um arquivo CSV."""
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col='model_name')
        return df.T.to_dict('list')
    return {}
