import pandas as pd

def load_pga_data(filepath: str) -> pd.DataFrame:
    """Carrega o dataset de PGA e pré-processa os IDs básicos e datas."""
    df = pd.read_csv(filepath)
    
    # Criação do ID único do terremoto (concatenação de data e tempo)
    df['earthquake_id'] = (df['date'].str.replace('-', '', regex=False) +
                           df['time'].str.replace(':', '', regex=False))
                           
    # Criação do ID que cruza estação e terremoto
    df['station_earthquakeid'] = df['station_key'] + '-' + df['earthquake_id']
    
    # Conversão da coluna de data para o tipo datetime do pandas
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def load_psa_data(filepath: str) -> pd.DataFrame:
    """Carrega o dataset de PSA (sinais espectrais)."""
    df = pd.read_csv(filepath)
    return df

def get_psa_station_earthquake_ids(df_psa: pd.DataFrame) -> set:
    """Extrai as chaves únicas 'station-earthquakeid' diretamente das colunas do dataset PSA."""
    station_earthquakeids = []
    for column in df_psa.columns:
        if column == 'period(s)':
            continue
        station = column.split('-')[0]
        eq_id = column.split('-')[1][:14]
        station_earthquakeids.append(f"{station}-{eq_id}")
    return set(station_earthquakeids)

def get_common_records(df_pga: pd.DataFrame, df_psa: pd.DataFrame) -> pd.DataFrame:
    """Filtra o DataFrame PGA mantendo apenas os registros presentes no DataFrame PSA."""
    psa_ids_set = get_psa_station_earthquake_ids(df_psa)
    
    df_pga_filtered = df_pga[df_pga['station_earthquakeid'].isin(psa_ids_set)].copy()
    df_pga_filtered.reset_index(drop=True, inplace=True)
    
    return df_pga_filtered