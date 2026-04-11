"""
Script de teste para validar o determinismo do pipeline de dados.

Garante que as etapas de divisão (split) de treino/teste e a 
clusterização (KMeans) gerem rigorosamente os mesmos resultados a
cada execução.
"""
import os
import pandas as pd

import config
from run_tuning import prepare_data
from main import set_global_seeds

def run_determinism_tests():
    print("Iniciando testes de determinismo do pipeline...\n")
    
    for cluster_col in config.CLUSTER_COLUMNS:
        print(f"{'='*60}")
        print(f"Testando a feature de cluster: {cluster_col}")
        print(f"{'='*60}")
        
        # Execução 1
        print("-> Execução 1 (Configurando sementes e processando dados...)")
        set_global_seeds(config.RANDOM_STATE)
        df_train_1, df_test_1 = prepare_data(cluster_col=cluster_col)
        
        # Execução 2
        print("-> Execução 2 (Resetando sementes e processando novamente...)")
        set_global_seeds(config.RANDOM_STATE)
        df_train_2, df_test_2 = prepare_data(cluster_col=cluster_col)
        
        print("\nComparando as saídas (pd.testing.assert_frame_equal)...")
        
        # Verificação do Conjunto de Treino
        try:
            pd.testing.assert_frame_equal(df_train_1, df_train_2)
            print(f"✅ SUCESSO (TREINO): O DataFrame é 100% idêntico (Determinístico).")
        except AssertionError as e:
            print(f"❌ ERRO (TREINO): Diferenças encontradas entre as execuções!")
            print(e)
            
        # Verificação do Conjunto de Teste
        try:
            pd.testing.assert_frame_equal(df_test_1, df_test_2)
            print(f"✅ SUCESSO (TESTE) : O DataFrame é 100% idêntico (Determinístico).\n")
        except AssertionError as e:
            print(f"❌ ERRO (TESTE): Diferenças encontradas entre as execuções!")
            print(e)

if __name__ == "__main__":
    run_determinism_tests()
