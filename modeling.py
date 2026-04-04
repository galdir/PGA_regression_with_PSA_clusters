import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import optuna
import keras_tuner as kt
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd

def get_preprocessor(num_attributes: list, cat_attributes: list = None) -> ColumnTransformer:
    """
    Cria o ColumnTransformer contendo as etapas de pré-processamento.
    Aplica StandardScaler para numéricos e OneHotEncoder para categóricos.
    """
    num_pipeline = Pipeline([
        ("standardize", StandardScaler()),
    ])
    
    transformers = [('num', num_pipeline, num_attributes)]
    
    if cat_attributes:
        cat_pipeline = make_pipeline(
            OneHotEncoder(handle_unknown="ignore")
        )
        # Inserindo o processador categórico no início da lista
        transformers.insert(0, ("cat", cat_pipeline, cat_attributes))
        
    return ColumnTransformer(transformers, remainder="drop")

def build_linear_regression(preprocessor: ColumnTransformer) -> Pipeline:
    """Constrói a pipeline para Regressão Linear."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor=LinearRegression(),
            func=np.log,
            inverse_func=np.exp
        ))
    ])

def build_polynomial_regression(preprocessor: ColumnTransformer, degree: int = 2, include_bias: bool = True) -> Pipeline:
    """Constrói a pipeline para Regressão Polinomial."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=degree, include_bias=include_bias)),
        ('model', TransformedTargetRegressor(
            regressor=LinearRegression(),
            func=np.log,
            inverse_func=np.exp
        ))
    ])

def build_elasticnet(preprocessor: ColumnTransformer, alpha: float = 0.1, l1_ratio: float = 0.5, 
                     max_iter: int = 100000, degree: int = 2, include_bias: bool = True) -> Pipeline:
    """Constrói a pipeline para Regressão Polinomial com regularização ElasticNet."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=degree, include_bias=include_bias)),
        ('model', TransformedTargetRegressor(
            regressor=ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter),
            func=np.log,
            inverse_func=np.exp
        ))
    ])

def build_random_forest(preprocessor: ColumnTransformer, n_estimators: int = 2100, 
                        max_depth: int = 14, min_samples_split: int = 9, 
                        min_samples_leaf: int = 4, max_features: float = 0.3) -> Pipeline:
    """Constrói a pipeline para Random Forest usando os melhores hiperparâmetros originais."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                max_features=max_features,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                oob_score=False,
                warm_start=False,
                random_state=42
            ),
            func=np.log,
            inverse_func=np.exp
        ))
    ])

def build_xgboost(preprocessor: ColumnTransformer, n_estimators: int = 1734, 
                  learning_rate: float = 0.2543527611706547, max_depth: int = 18, 
                  subsample: float = 0.5809548106071651, colsample_bytree: float = 0.8882288814017651, 
                  min_child_weight: int = 1, gamma: float = 8.804203083872113e-07, 
                  reg_alpha: float = 0.272597395385441, reg_lambda: float = 0.10188365398056981) -> Pipeline:
    """Constrói a pipeline para XGBoost usando os melhores hiperparâmetros originais."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor=xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42
            ),
            func=np.log,
            inverse_func=np.exp
        ))
    ])

def build_dnn_model(n_hidden_layers: int = 3, n_neurons: int = 419, 
                    activation: str = "swish", learning_rate: float = 0.0005656677780014923,
                    dropout_rate: float = 0.0) -> Sequential:
    """
    Constrói e compila o modelo Keras (Deep Neural Network) 
    usando os melhores hiperparâmetros encontrados pelo Keras Tuner.
    """
    model = Sequential()
    for _ in range(n_hidden_layers):
        model.add(Dense(n_neurons, activation=activation, kernel_initializer="he_normal"))
        model.add(BatchNormalization())
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    rmse_metric = tf.keras.metrics.RootMeanSquaredError(name='rmse')
    nn_optimizer = keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True, momentum=0.9)
    
    model.compile(loss="huber", optimizer=nn_optimizer, metrics=[rmse_metric])
    return model


# ==============================================================================
# HYPERPARAMETER TUNING FUNCTIONS
# ==============================================================================

def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series, preprocessor: ColumnTransformer, 
                       n_trials: int = 300, random_state: int = 42):
    """
    Executa a busca de hiperparâmetros para o Random Forest usando Optuna e GroupKFold
    (agrupando pelos IDs dos terremotos para evitar Data Leakage).
    """
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'random_state': random_state,
            'n_jobs': -1
        }

        reg = TransformedTargetRegressor(
            regressor=RandomForestRegressor(**param),
            func=np.log,
            inverse_func=np.exp
        )
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', reg)
        ])

        gkf = GroupKFold(n_splits=5)
        score = cross_val_score(pipeline, X_train, y_train, cv=gkf, groups=groups,
                                scoring='neg_root_mean_squared_error').mean()
        return score

    # Adiciona o TPESampler com a semente para garantir reprodutibilidade na escolha dos parâmetros
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    return study


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series, 
                 preprocessor: ColumnTransformer, n_trials: int = 300, random_state: int = 42):
    """
    Executa a busca de hiperparâmetros para o XGBoost usando Optuna e GroupKFold 
    (agrupando pelos IDs dos terremotos para evitar Data Leakage).
    """
    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 2100),
            'learning_rate': trial.suggest_float('learning_rate',  1e-4, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 20), # Reduzido de 30 para 20 para evitar explosão de RAM
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 0, 25),
            'gamma': trial.suggest_float('gamma', 1e-8, 5, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 1e2, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 1e2, log=True),
            'random_state': random_state,
            'n_jobs': -1,
            'early_stopping_rounds': 50 # Habilita parada antecipada
        }

        gkf = GroupKFold(n_splits=5)
        rmses = []
        
        # Loop manual permite usar eval_set (Early Stopping) e Pruning (Optuna)
        for step, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
            X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            # Evita Data Leakage processando separadamente
            X_train_processed = preprocessor.fit_transform(X_train_fold)
            X_val_processed = preprocessor.transform(X_val_fold)
            
            # Aplica transformação logarítmica do Target
            y_train_fold_log = np.log(y_train_fold)
            y_val_fold_log = np.log(y_val_fold)
            
            model = xgb.XGBRegressor(**param)
            model.fit(X_train_processed, y_train_fold_log, 
                      eval_set=[(X_val_processed, y_val_fold_log)], verbose=False)
            
            # Calcula o RMSE no espaço original (após inverter o log com np.exp)
            preds = np.exp(model.predict(X_val_processed))
            rmses.append(root_mean_squared_error(y_val_fold, preds))
            
            # Reporta a média atual para o Optuna; aborta a Trial se for ruim demais
            trial.report(-np.mean(rmses), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return -np.mean(rmses)

    # Adiciona Sampler determinístico e Pruner (ignora as primeiras 5 trials do estudo antes de começar a podar)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    
    return study


def tune_dnn(X_train, y_train, X_valid, y_valid, preprocessor: ColumnTransformer, 
             max_epochs: int = 50, project_name: str = 'keras_tuner'):
    """
    Executa a busca de hiperparâmetros para a Deep Neural Network usando Keras Tuner.
    """
    def build_model(hp):
        n_hidden = hp.Int("n_hidden", min_value=1, max_value=11, default=2)
        n_neurons = hp.Int("n_neurons", min_value=10, max_value=1800)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        optimizer_choice = hp.Choice("optimizer", values=["adam", "nesterov", "adamw", "sgd"])
        activation_function = hp.Choice("activation", values=["relu", "swish"])
        dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)

        if optimizer_choice == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_choice == "nesterov":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True, momentum=0.9)
        elif optimizer_choice == "adamw":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())

        for _ in range(n_hidden):
            model.add(tf.keras.layers.Dense(n_neurons, activation=activation_function, kernel_initializer="he_normal"))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(1))
        model.compile(loss="huber", optimizer=optimizer, metrics=["root_mean_squared_error"])
        return model

    tf_x_train = preprocessor.fit_transform(X_train)
    tf_x_valid = preprocessor.transform(X_valid)
    
    y_train_log = np.log(y_train)
    y_valid_log = np.log(y_valid)

    tuner = kt.Hyperband(build_model,
                         objective=kt.Objective("val_root_mean_squared_error", direction="min"),
                         max_epochs=max_epochs,
                         factor=3,
                         directory='tuning_results',
                         project_name=project_name,
                         seed=42,
                         overwrite=True)
                         
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_root_mean_squared_error', 
        patience=10,
        restore_best_weights=True
    )
    
    tuner.search(tf_x_train, y_train_log, validation_data=(tf_x_valid, y_valid_log), 
                 callbacks=[stop_early], verbose=2)
                 
    return tuner