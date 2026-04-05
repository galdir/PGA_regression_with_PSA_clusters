# Plano de Implementação: Tuning Específico por Atributo de Cluster

## Objetivo
Habilitar a busca de hiperparâmetros otimizados especificamente para modelos que utilizam atributos de clusterização (KMeans sobre dados de PSA), mantendo a capacidade de rodar o pipeline atual (que utiliza os hiperparâmetros do modelo base) através de uma flag de configuração.
Continuaremos avaliando os 4 tipos de clusterização (`clusters_spectral_coefs`, `clusters_spectral_means`, `clusters_spectral_horiz_means`, `clusters_spectral_vert_means`).

---

## Passo 1: Atualização das Configurações (`config.py`)
Adicionar uma nova flag booleana para controlar o comportamento do `main.py`.

* **Modificação:** 
  Adicionar `USE_CLUSTER_SPECIFIC_TUNING = False` na seção de flags de controle de execução.
  Se `False`, o `main.py` usará os parâmetros base (comportamento atual).
  Se `True`, o `main.py` buscará os arquivos `.json` específicos de cada cluster.

---

## Passo 2: Adaptação do Script de Otimização (`run_tuning.py`)
Atualmente, o `run_tuning.py` otimiza apenas com as features numéricas base, pois ele não executa o pipeline de clusterização no `prepare_data()`. Precisaremos adaptá-lo para que ele saiba como afinar modelos com os clusters.

* **Modificação 1 (Geração de Dados):**
  Atualizar a função `prepare_data()` no `run_tuning.py` para rodar o cálculo de médias espectrais (`calculate_psa_means`) e geração de clusters (`generate_clusters`), anexando os rótulos de cluster ao conjunto `df_train_clean`.
  
* **Modificação 2 (Argumentos CLI):**
  Adicionar um argumento `--cluster` no `argparse` com opções: `none` (padrão), ou um dos nomes exatos das features de cluster.
  - Se `none`: Roda o tuning igual hoje (modelo base), salvando `tuning_results_{model}.json`.
  - Se um nome de cluster for fornecido (ex: `clusters_spectral_means`): Otimiza o modelo obrigatoriamente para esse único atributo de cluster.

* **Modificação 3 (Pipeline e Pré-processamento):**
  Se um cluster for especificado, passar o atributo de cluster para a função `get_preprocessor(num_attributes=..., cat_attributes=[nome_do_cluster])` para que o `OneHotEncoder` seja ativado durante a busca.

* **Modificação 4 (Arquivos de Saída):**
  Salvar os resultados com o padrão de nomenclatura estendido: 
  `tuning_results_{model}_{nome_do_cluster}.json` (ex: `tuning_results_rf_clusters_spectral_means.json`).

---

## Passo 3: Adaptação do Pipeline Principal (`main.py`)
O `main.py` precisa ler a nova flag e decidir quais hiperparâmetros instanciar dentro do loop de teste de modelos de cluster.

* **Modificação 1 (Leitura de Json Dinâmica):**
  Alterar o loop de treinamento atual:
  ```python
  for cluster_col in config.CLUSTER_COLUMNS:
      ...
  ```
  Para adicionar uma verificação de parâmetros:
  ```python
  if config.USE_CLUSTER_SPECIFIC_TUNING:
      # Tenta buscar ex: tuning_results_rf_clusters_spectral_means.json
      rf_params_cluster = get_tuned_params(f"rf_{cluster_col}")
      xgb_params_cluster = get_tuned_params(f"xgb_{cluster_col}")
      dnn_params_cluster = get_tuned_params(f"dnn_{cluster_col}")
      
      # Fallback caso o arquivo não exista (ex: o usuário ativou a flag mas não rodou o script de tuning)
      if not rf_params_cluster: rf_params_cluster = rf_params
      # ... repete para os demais
  else:
      # Comportamento antigo: usa o base
      rf_params_cluster = rf_params
      xgb_params_cluster = xgb_params
      dnn_params_cluster = dnn_params
  ```

* **Modificação 2 (Instanciação com Novos Parâmetros):**
  Nas chamadas `build_random_forest(..., **rf_params_cluster)`, `build_xgboost`, e `train_evaluate_dnn`, usar a variável com sufixo `_cluster` ao invés da genérica.

---

## Passo 4: Fluxo de Execução Recomendado (Quando Implementado)

1. Deixar `config.USE_CLUSTER_SPECIFIC_TUNING = False`.
2. Rodar o tuning base (caso ainda não tenha os melhores parâmetros base): `python run_tuning.py --model rf` (o padrão já será `none`).
3. Rodar o tuning focado no cluster de interesse: `python run_tuning.py --model rf --cluster clusters_spectral_means`.
4. Repetir o passo 3 para `xgb` e `dnn`.
5. Trocar `config.USE_CLUSTER_SPECIFIC_TUNING = True`.
6. Executar o pipeline analítico: `python main.py`.

Os resultados consolidados agora mostrarão a comparação real entre o "Melhor Modelo Base" vs "Melhor Modelo Específico de Cada Cluster".
