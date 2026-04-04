# Plano de Refatoração: PGA Regression with PSA Clusters

## Visão Geral
O código atual é um script monolítico (mais de 2.000 linhas) gerado a partir de um Jupyter Notebook. Ele mistura carregamento de dados, análise exploratória (EDA), pré-processamento, clusterização, treinamento de dezenas de modelos e plotagem de gráficos em um único escopo global.

O objetivo desta refatoração é modularizar o código, facilitando a manutenção, o reaproveitamento de componentes e a legibilidade.

**Regra de Ouro:** O arquivo original (`peak_ground_acceleration_regression_with_psa_clusters_pre_split.py`) **NÃO** deve ser alterado em nenhuma hipótese. Além disso, a refatoração deve ter como prioridade máxima **reproduzir fielmente os resultados originais**, mantendo a mesma ordem de execução, lógicas de separação e cálculos matemáticos do notebook. Todo o trabalho de refatoração deve focar exclusivamente na criação de **novos arquivos** e na extração/cópia da lógica para eles, mantendo o monolito intacto.

---

## Passo 1: Extração de Configurações (Configuration Management)
Atualmente, URLs, parâmetros de modelos e *flags* de execução (ex: `load_trained_models`, `hiperparameters_tuning`) estão espalhados pelo código.

* **Ação:** Criar um arquivo `config.py` (ou `config.yaml`).
* **O que extrair (copiar do original):** 
  * URLs de download dos dados (ex: `raw_earthquakes_pga_url`).
  * Caminhos de diretórios para salvar/carregar modelos (`./paper_1_results_revised/`).
  * Flags booleanas de controle de fluxo.
  * Definição de listas de colunas e *features* (ex: `selected_atributes`, `num_attributes`).

## Passo 2: Modularização por Responsabilidade
Criar novos módulos `.py` baseados na lógica do arquivo monolítico, cada um focado em uma única responsabilidade. O novo projeto deve ter a seguinte estrutura de arquivos:

1. **`data_loader.py`**
   * Funções para fazer o download, descompactar e carregar os arquivos CSV e SHP na memória.
   * Função para cruzar e validar quais registros existem tanto em PGA quanto em PSA.

2. **`preprocessing.py`**
   * Funções para criação de novas variáveis (Feature Engineering, ex: `log_incidence_angle`, categorias de profundidade/magnitude) aplicando as transformações numéricas exatas do código original.
   * Funções para separar dados de Treino e Teste replicando perfeitamente a lógica cumulativa de agrupamento por `earthquake_id` do original.
   * Remoção de *outliers* utilizando o `StandardScaler` seguido do `LocalOutlierFactor` (ajustado e aplicado apenas no conjunto de treino).

3. **`clustering.py`**
   * Processamento das matrizes espectrais (PSA).
   * Cálculo das médias verticais, horizontais e coeficientes.
   * Execução do algoritmo `KMeans` e mapeamento dos *labels* de volta para os *DataFrames*.

4. **`modeling.py`**
   * Definição das *Pipelines* do *Scikit-Learn* e pré-processadores numéricos/categóricos.
   * Funções para busca de hiperparâmetros (*GridSearch*, *Optuna*, *Keras Tuner*).
   * Construção da rede neural profunda com *TensorFlow/Keras*.

5. **`evaluation.py`**
   * Rotinas como `model_experiment()` que calculam métricas (RMSE, R2, Intervalos de Confiança).
   * Funções para salvar e carregar os pipelines treinados (`joblib` / `pickle`).

6. **`visualization.py`**
   * Centralizar toda a lógica envolvendo `matplotlib`, `seaborn` e `cartopy`.
   * Exemplos: Mapas de epicentros, curvas espectrais, gráficos de dispersão de erros e distribuições (histogramas).

## Passo 3: Criação de um Ponto de Entrada (Entry Point)
Criar um arquivo orquestrador, por exemplo, `main.py` ou `run_pipeline.py`.

* **Ação:** Este script não deve conter lógica complexa, apenas importar funções dos módulos criados no *Passo 2* e executá-las em sequência.
* **Benefício:** Um desenvolvedor conseguirá ler o `main.py` de cima a baixo e entender exatamente as etapas do pipeline sem se perder em detalhes de implementação de gráficos ou *loops* extensos.

## Passo 4: Eliminação de Variáveis Globais e Isolamento de Escopo
Hoje o código original manipula *DataFrames* globalmente (ex: alterar `df` afeta tudo daqui para baixo).

* **Ação:** Nos novos arquivos criados, encapsular todo o código solto em funções (`def`).
* **Regra:** *DataFrames* e modelos devem ser passados como argumentos para as funções, e os resultados devem ser retornados (usando o princípio de *pure functions* sempre que possível).

## Passo 5: Limpeza de Código Morto / Comentado
O script original possui extensos blocos de código comentados (`# xbg_n_estimators = ...`, testes antigos do Optuna, etc.). 
* **Ação:** Ignorar esses blocos ao copiar a lógica para os novos arquivos. O novo código (`main.py` e módulos) deve ser limpo e livre de experimentos mortos, deixando-os apenas no script original como registro.