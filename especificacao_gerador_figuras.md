# Especificação: Gerador de Figuras (Análise Visual)

## 1. Visão Geral
Este documento especifica os requisitos para um novo script Python (ex: `generate_figures.py`) dedicado exclusivamente à geração das figuras e mapas para análise exploratória e resultados do projeto.
O objetivo é extrair essa responsabilidade do script de orquestração (`main.py`) e do notebook original, mantendo o código modular e organizado.

## 2. Entradas Necessárias
Para gerar todas as figuras, o script precisará carregar:
1. Dados brutos (`earthquakes_pga.csv` e `earthquakes_psa_earthquakeid.csv`).
2. *Shapefiles* do México para plotagem de mapas (`geopandas`).
3. As saídas de pré-processamento (datasets de treino e teste já divididos e limpos, além das features geradas).
4. O dicionário de agrupamentos (clusters).
5. Os modelos/pipelines treinados que foram salvos em disco (para gerar as previsões e comparar com o PGA medido).

## 3. Diretório de Saída
As figuras deverão ser salvas em um diretório específico (ex: `./paper_1_results_revised/figures/`) com alta resolução (DPI adequado para publicação).

## 4. Lista de Figuras a Serem Geradas

O script deverá conter funções modulares para gerar exatamente as seguintes visualizações:

### 4.1. Análise Exploratória de Dados (EDA)
* **Figura 1: Mapa do Conjunto de Dados**
  * **Descrição:** Localização das estações sísmicas (círculos pretos) e dos epicentros dos terremotos (estrelas) no mapa do México.
  * **Detalhes:** A cor e o tamanho das estrelas devem indicar a magnitude dos terremotos.
* **Figura 2: Histogramas de Atributos**
  * **Descrição:** Matriz de subplots (ex: 4x3) com histogramas dos principais atributos do conjunto de dados.
  * **Detalhes:** Deve incluir uma linha contínua indicando a estimativa de densidade do kernel (`kde=True`).
* **Figura 3: Estudo de Caso de Evento Específico**
  * **Descrição:** Mapa com foco no terremoto de 12 (ou 19) de setembro de 2017.
  * **Detalhes:** Mostrar diversas estações localizadas a distâncias comparáveis do epicentro, posicionadas próximas umas das outras, evidenciando as variações notáveis na aceleração máxima do solo (PGA) registrada (tamanho/cor dos marcadores das estações baseados no PGA).

### 4.2. Análise de Sinais PSA (Pseudo-Spectral Accelerations)
* **Figura 4: Semelhança Intra-estação de PSAs**
  * **Descrição:** Subplots lado a lado mostrando PSAs de múltiplas ocorrências sísmicas.
  * **Detalhes:** À esquerda, PSAs da estação CALE; à direita, PSAs da estação ACAD. Objetivo: evidenciar a semelhança de formato dos sinais pertencentes à mesma estação independente do evento.
* **Figura 5: Vetores de Pré-processamento**
  * **Descrição:** Plot 2x2 mostrando o "sinal médio" ou "vetor de features" extraído para as estações.
  * **Detalhes:** (a) Razão Espectral; (b) Espectros de Três Eixos; (c) Espectros Horizontais; (d) Espectros Verticais.

### 4.3. Análise de Clusterização
* **Figura 6: Avaliação de Silhueta (Three Axis Spectra)**
  * **Descrição:** Gráficos da pontuação de silhueta (`SilhouetteVisualizer`).
  * **Detalhes:** Avaliação de Ks (ex: 2 a 5) a partir do método de agrupamento de espectros de três eixos.
* **Figura 7: Agrupamentos KMeans - Espectros de Três Eixos**
  * **Descrição:** Sinais agrupados em subplots (um por cluster). Sinais em cinza, média em vermelho.
* **Figura 8: Agrupamentos KMeans - Razão Espectral**
  * **Descrição:** Sinais agrupados resultantes dos vetores de Razão Espectral.
* **Figura 9: Agrupamentos KMeans - Espectros Horizontais**
  * **Descrição:** Sinais agrupados resultantes dos vetores de Espectros Horizontais.
* **Figura 10: Agrupamentos KMeans - Espectros Verticais**
  * **Descrição:** Sinais agrupados resultantes dos vetores de Espectros Verticais.

### 4.4. Resultados de Modelagem e Regressão
* **Figura 11: PGA Medido vs Previsto (Baseline / Sem Cluster PSA)**
  * **Descrição:** Gráficos de dispersão (Scatter plot em escala log) comparando os valores reais e previstos pelos modelos base (Random Forest, XGBoost, DNN, etc.).
  * **Detalhes:** Incluir a reta de identidade (y=x) pontilhada como referência.
* **Figura 12: PGA Medido vs Previsto (Com Cluster - Espectros Horizontais)**
  * **Descrição:** Idêntico à Figura 11, mas utilizando as previsões do modelo treinado com o atributo de clusterização de *Média de Espectros Horizontais*.

### 4.5. Regionalização e Simulação
* **Figura 13: Mapa de Regionalização por Cluster**
  * **Descrição:** Mapa com a localização das estações sísmicas no México.
  * **Detalhes:** A cor dos círculos deve indicar explicitamente o cluster de *Média de Espectros Horizontais* a qual a estação pertence.
* **Figura 14: Simulação de PGA por Cluster (Terremoto Hipotético)**
  * **Descrição:** Gráfico de barras demonstrando o impacto direto do cluster na previsão de PGA utilizando o modelo vencedor dos experimentos (**Random Forest**).
  * **Detalhes:** PGA previsto para cada possível agrupamento de *média de espectros horizontais* (0, 1, 2, 3...) mantendo todas as outras variáveis (distância, profundidade, magnitude, localização) constantes para um mesmo evento simulado. As previsões devem ser extraídas especificamente do pipeline treinado `Random Forest with PSA clusters_spectral_horiz_means`.

## 5. Estrutura Proposta do Script
* `load_data_and_models()`: Rotina para restaurar o estado dos dados e modelos salvos.
* `plot_eda_figures()`: Agrupa funções para as Figuras 1 a 3.
* `plot_psa_signals()`: Agrupa funções para as Figuras 4 e 5.
* `plot_clustering_results()`: Agrupa funções para as Figuras 6 a 10.
* `plot_regression_results()`: Agrupa funções para as Figuras 11 e 12.
* `plot_regionalization_and_simulation()`: Agrupa funções para as Figuras 13 e 14.
* `main()`: Orquestra a geração e salva todas as figuras na pasta de resultados.