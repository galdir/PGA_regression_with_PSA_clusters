# Sugestões de Aprimoramento do Projeto PGA vs PSA

Este documento lista oportunidades arquiteturais, estatísticas e de modelagem para aprimorar os resultados (redução de RMSE e aumento de R²) do projeto de previsão de Peak Ground Acceleration (PGA) utilizando clusterização de dados de PSA.

## 1. Transformação da Variável Alvo (Target)
* **Problema:** A aceleração de pico (PGA) geralmente segue uma distribuição log-normal, variando em várias ordens de grandeza. O treinamento com os valores brutos faz com que os modelos de Machine Learning foquem excessivamente nos valores absolutos extremos e permite previsões de valores negativos (o que é fisicamente impossível).
* **Solução:** Treinar os modelos para prever o **logaritmo natural do PGA** (`np.log(PGA)`). Equações tradicionais de previsão de movimento do solo (GMPEs) baseiam-se em `ln(PGA)`.
* **Como implementar:** Usar o `TransformedTargetRegressor` do Scikit-Learn. Ele aplica o logaritmo no treinamento de forma transparente e aplica a exponencial inversa (`np.exp`) na inferência.


## 2. Uso de k especifico por tecnica de preprocessamento de clustering

## 3. busca de hiperparametros usando atributos de clusterização

## 4. Aprimoramento da Clusterização de PSA (Soft Clustering)
* **Distância aos Centróides:** Atualmente, usa-se o algoritmo KMeans para gerar um único rótulo categórico (`K=4`). No entanto, os perfis de solo na vida real são misturas contínuas. Em vez de usar o rótulo binário (One-Hot Encoding), usar as **distâncias espaciais para cada um dos 4 centróides** fornecerá 4 features numéricas contínuas. Isso oferece ao modelo de regressão uma representação muito mais sutil e rica do perfil espectral do solo.
* **Otimização do K:** O número de clusters (K=4) está fixo. Pode-se implementar a otimização do `K` como um hiperparâmetro utilizando a métrica *Silhouette Score* ou via Optuna.
* **Algoritmos Alternativos:** Experimentar *Gaussian Mixture Models (GMM)* em vez do KMeans, pois eles lidam matematicamente melhor com a probabilidade de um solo pertencer a mais de um agrupamento espectral.

## 5. Aproveitamento de Features Descartadas
* **Features Categóricas Criadas:** Em `preprocessing.py`, são criadas colunas interessantes como `magnitude_cat`, `calculated_epicentral_distance_cat` e `depth_cat`. Porém, observando o `MODEL_NUM_FEATURES`, elas não são passadas aos modelos Scikit-Learn. Incluir essas variáveis categorizadas via One-Hot Encoding ajuda modelos paramétricos (como Regressão Linear/ElasticNet) a capturarem não-linearidades espaciais.
* **Dados Espaciais (Coordenadas):** Latitude e Longitude não são utilizadas na predição final. Inserir essas coordenadas permite que algoritmos baseados em árvore (Random Forest, XGBoost) regionalizem as previsões e modelem implicitamente os padrões geológicos de determinadas bacias ou regiões.

## 6. Estratégia de Modelagem e Treinamento
* **Stacking / Ensembling:** Como o projeto já treina ótimos modelos base (RF, XGBoost, DNN), o próximo passo natural é juntá-los. Um `StackingRegressor` usando a previsão dos 3 modelos combinados (com um meta-modelo linear) pode oferecer uma redução drástica e robusta no RMSE.
* **Novos Algoritmos (CatBoost / LightGBM):** O CatBoost é excepcionalmente forte em problemas que possuem features numéricas contínuas misturadas com variáveis estritamente categóricas (como o ID do cluster). Muitas vezes, ele atinge scores superiores ao XGBoost sem necessidade de `OneHotEncoder`.
* **Pesos por Amostra (Sample Weighting):** Terremotos altamente destrutivos (Alto PGA/Magnitude) são os eventos mais críticos na engenharia civil, mas são os mais raros no dataset. Pode-se atribuir um peso maior às amostras com maior magnitude durante a chamada `.fit(sample_weight=pesos)` para forçar o modelo a "errar menos" nesses eventos destrutivos em vez de focar nos terremotos pequenos e comuns.

## 7. Cuidado com a Remoção de Outliers (Local Outlier Factor)
* **Contexto Sismológico:** Na engenharia sísmica, um valor anômalo de aceleração num sensor nem sempre é um erro instrumental. Muitas vezes é o resultado de direcionalidade de ruptura da falha, efeitos locais de sítio extremos, ou rupturas rasas incomuns — exatamente os eventos perigosos que o modelo deve tentar antecipar.
* **Sugestão:** Executar o pipeline (com Optuna) desabilitando temporariamente o `remove_outliers` no conjunto de treino. Ao remover registros extremos da distribuição de treino, podemos estar impedindo o XGBoost e a Random Forest de aprender as regras que preveem os tremores mais atípicos do conjunto de teste.



## 8. Hipotese: usar gmm no clustering

## 9. Hipotese: Tratar o "K" como um Hiperparâmetro (A abordagem ideal) A