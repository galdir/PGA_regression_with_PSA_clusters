# Especificação de Arquitetura e Pipeline de Dados

## 1. Visão Geral do Projeto
**Objetivo:** Prever a Aceleração de Pico no Solo (PGA - *Peak Ground Acceleration*) utilizando algoritmos de Machine Learning. O projeto testa a hipótese principal de que a introdução de atributos gerados por clusterização de perfis de Aceleração Pseudo-Espectral (PSA) reduz significativamente o erro das predições, suprindo a ausência de dados geológicos detalhados do solo.

## 2. Fontes de Dados
* **PGA Data (`earthquakes_pga.csv`)**: Contém informações sobre o terremoto, estações sísmicas, distâncias e o registro do PGA.
* **PSA Data (`earthquakes_psa_earthquakeid.csv`)**: Contém registros espectrais nos eixos Vertical (V), Leste-Oeste (EW) e Norte-Sul (NS).
* **Chave de Cruzamento**: Os datasets são cruzados usando a chave composta `station_earthquakeid` (Estação + ID do Terremoto gerado por Data/Hora).

## 3. Pipeline de Processamento (Orquestrado por `main.py`)
O fluxo de execução segue etapas estritas para garantir robustez e evitar vazamento de dados (*Data Leakage*):

### 3.1. Feature Engineering
* Cálculo do logaritmo do ângulo de incidência (`log_incidence_angle`).
* Categorização (Binning) de variáveis contínuas (Magnitude, Distância Epicentral, Profundidade, PGA). *(Nota: Atualmente geradas, mas não passadas aos modelos Scikit-Learn).*

### 3.2. Separação Treino / Teste (Data Split)
* **Proporção**: 80% Treino / 20% Teste.
* **Agrupamento**: A divisão é agrupada pelo `earthquake_id`.
* **Motivo**: Prevenir que registros do mesmo terremoto, captados por estações diferentes, existam simultaneamente no Treino e no Teste, o que causaria um viés otimista (*data leakage* espacial e temporal).

### 3.3. Tratamento de Outliers
* **Algoritmo**: Local Outlier Factor (LOF) com 200 vizinhos (`n_neighbors=200`) e contaminação de 0.001.
* **Aplicação**: Aplicado **exclusivamente no conjunto de treinamento**. Outliers não são removidos do teste para garantir que a avaliação reflita a realidade completa do domínio.

### 3.4. Processamento de Sinais PSA e Clusterização
Os sinais de PSA são divididos garantindo que o Teste não influencie o Treino:
1. **Extração Espectral**: São calculadas médias para os sinais de cada estação nas direções V, EW e NS.
2. **Geração de Componentes**:
   * `spectral_coefs`: Coeficientes espectrais (EW/V e NS/V).
   * `spectral_means`: Médias espectrais considerando os 3 eixos.
   * `spectral_horiz_means`: Médias horizontais (EW e NS).
   * `spectral_vert_means`: Médias apenas do eixo vertical (V).
3. **Clusterização (KMeans)**: 
   * Aplicada usando `MinMaxScaler` prévio.
   * `K` (Número de Clusters) pré-definido pelo Método do Cotovelo / Silhouette:
     * `spectral_coefs`: K = 5
     * `spectral_means`: K = 4
     * `spectral_horiz_means`: K = 4
     * `spectral_vert_means`: K = 3
   * Modelos são "fitados" no Treino e as etiquetas preditas no Teste via `.predict()`.

## 4. Avaliação e Resultados
* Os pipelines são salvos localmente (`.joblib` para Scikit-Learn / XGBoost, `.keras` para modelos profundos).
* O sistema gera gráficos de dispersão (Medido vs. Previsto) em escala logarítmica.
* Análises de erro absoluto/percentual comparam diretamente os modelos **Base** (Sem PSA) contra os modelos **Com PSA**.
* **Reprodutibilidade**: Seeds fixadas globalmente (`config.RANDOM_STATE = 42`) para bibliotecas padrões, Numpy e TensorFlow.