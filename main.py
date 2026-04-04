import config
from data_loader import load_pga_data, load_psa_data, get_common_records, get_psa_station_earthquake_ids
from preprocessing import split_train_test_by_earthquake, create_features, remove_outliers
from clustering import split_psa_columns, calculate_psa_means, generate_clusters, add_clusters

def main():
    print("1. Carregando os dados...")
    df_pga = load_pga_data(config.RAW_EARTHQUAKES_PGA_URL)
    df_psa = load_psa_data(config.RAW_EARTHQUAKES_PSA_URL)
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
        features=config.SELECTED_ATTRIBUTES, 
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

    print("\n--- Pipeline inicial concluída com sucesso! ---")

if __name__ == "__main__":
    main()