# Especificação de Modelagem (Machine Learning)

## 1. Definição do Problema e Variáveis
* **Tipo de Tarefa**: Regressão.
* **Variável Alvo (Target)**: `peak_ground_acceleration` (PGA).
  * **Transformação do Alvo**: Todos os algoritmos usam `TransformedTargetRegressor` aplicando uma transformação Logarítmica (`np.log`) na entrada do treino e Exponencial (`np.exp`) na saída das predições.

### 1.1. Features Utilizadas (Input)
**Numéricas (Base):**
* `magnitude`
* `calculated_epicentral_distance`
* `depth`
* `log_incidence_angle`
* `calculated_hypocentral_distance`
* `source_off_shore`

**Categóricas (Features de Estudo / Cluster PSA):**
* Avaliadas individualmente em pipelines separados:
  * `clusters_spectral_coefs`
  * `clusters_spectral_means`
  * `clusters_spectral_horiz_means`
  * `clusters_spectral_vert_means`

## 2. Pipeline de Pré-processamento
* **Numéricas**: Padronização através do `StandardScaler`.
* **Categóricas**: Codificação com `OneHotEncoder(handle_unknown="ignore")`.
* *Nota: Regressão Polinomial e ElasticNet aplicam um passo extra de `PolynomialFeatures(degree=2)` seguido de novo `StandardScaler` apenas na via numérica.*

## 3. Algoritmos Avaliados
1. **Linear Regression**
2. **Polynomial Regression** (Regressão Linear com expansão polinomial de grau 2)
3. **ElasticNet** (Regressão Polinomial + Regularização L1/L2)
4. **Random Forest Regressor**
5. **XGBoost Regressor** (Métrica interna: `reg:squarederror`)
6. **Deep Neural Network (DNN)**: Keras Sequential (Camadas Densas + BatchNormalization + Dropout opcional, ativador ReLU ou Swish).

## 4. Otimização de Hiperparâmetros (Tuning)
O ajuste de hiperparâmetros foi implementado de forma a proteger contra vazamento de dados, respeitando o agrupamento temporal/espacial dos terremotos.

### 4.1. Estratégia de Validação
* **XGBoost e Random Forest**: Optuna aliado a `GroupKFold(n_splits=5)`. O `groups` é definido pela coluna `earthquake_id`.
* **Deep Neural Network**: *Keras Tuner* (`Hyperband`) utilizando validação fixa (*Holdout*) a partir de um split limpo através de `GroupShuffleSplit`.
* **Métrica Objetivo**: Maximização do RMSE negativo (Scikit-Learn/Optuna) ou minimização do `val_root_mean_squared_error` (Keras Tuner).

### 4.2. Detalhes de Busca e *Early Stopping*
* **XGBoost**:
  * Utiliza Pruning (`MedianPruner`) no Optuna para interromper *trials* ruins baseando-se no desempenho em cada dobra (*fold*).
  * Avaliação do RMSE é feita no espaço original da variável, revertendo a escala logarítmica com `np.exp` no momento do cálculo do erro do Optuna.
* **DNN**:
  * Callbacks: `EarlyStopping` monitorando o RMSE de validação com paciência de 10 épocas, restaurando os melhores pesos.
  * Otimizadores testados: Adam, AdamW, Nesterov, SGD.

## 5. Avaliação do Modelo (Evaluation)
Métricas acompanhadas para o conjunto de testes:
* **RMSE (Root Mean Squared Error)** (Principal métrica de ranqueamento da tabela de resultados).
* **R² (Coeficiente de Determinação)**.
* **Intervalo de Confiança (95%)**: Calculado estatisticamente (SciPy) usando a variância dos erros quadráticos das predições do conjunto de teste contra os valores reais.