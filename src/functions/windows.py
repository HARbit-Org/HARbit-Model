from .frequency_domain_features import extract_frequency_domain_features
from .time_domain_features import extract_time_domain_features

import polars as pl
import pandas as pd
import numpy as np


def create_feature_windows(df, window_seconds=5, overlap_percent=50, sampling_rate=20):
    """
    Crea ventanas con extracción completa de características
    """
    # Calcular parámetros de ventana
    step_seconds = window_seconds * (100 - overlap_percent) / 100
    step_duration_str = f"{int(step_seconds * 1000)}ms"
    window_duration_str = f"{int(window_seconds * 1000)}ms"
    
    # Crear ventanas usando group_by_dynamic
    windows = df.group_by_dynamic(
        index_column='Timestamp',
        every=step_duration_str,
        period=window_duration_str,
        closed='left',
        group_by=['Subject-id', 'Activity Label']
    ).agg([
        pl.col('X').alias('X_data'),
        pl.col('Y').alias('Y_data'),
        pl.col('Z').alias('Z_data'),
        pl.col('Timestamp').min().alias('window_start'),
        pl.col('Timestamp').max().alias('window_end'),
        pl.count().alias('sample_count')
    ]).filter(pl.col('sample_count') >= window_seconds * sampling_rate * 0.8)  # Filtrar ventanas incompletas
    
    # Convertir a pandas para el procesamiento de características
    windows_pd = windows.to_pandas()
    
    # Lista para almacenar todas las características
    feature_list = []
    
    for idx, row in windows_pd.iterrows():
        # Crear DataFrame de la ventana
        window_data = pd.DataFrame({
            'X': row['X_data'],
            'Y': row['Y_data'],
            'Z': row['Z_data']
        })
        # print(window_data.shape)
        
        # Extraer características
        time_features = extract_time_domain_features(window_data)
        freq_features = extract_frequency_domain_features(window_data, sampling_rate)
        
        # Combinar todas las características
        all_features = {
            'Subject-id': row['Subject-id'],
            'Activity Label': row['Activity Label'],
            'window_start': row['window_start'],
            'window_end': row['window_end'],
            'sample_count': row['sample_count'],
            **time_features,
            **freq_features
        }
        
        feature_list.append(all_features)
    
    return pd.DataFrame(feature_list)