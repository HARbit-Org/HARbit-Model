import pandas as pd
import polars as pl
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler
)
from sklearn.model_selection import train_test_split

def prepare_features_for_cnn_lstm_direct(features_df):
    """
    Prepara características ya extraídas DIRECTAMENTE para CNN-LSTM
    Sin crear ventanas adicionales - cada fila es una muestra independiente
    """
    # Convertir a pandas si es necesario
    if hasattr(features_df, 'to_pandas'):
        features_df = features_df.to_pandas()

    # Identificar columnas de características
    metadata_cols = ['Subject-id', 'Activity Label', 'window_start', 'window_end', 'sample_count']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    print(f"Características detectadas: {len(feature_cols)}")
    print(f"Muestras totales: {len(features_df)}")
    
    # Extraer características y etiquetas directamente
    X = features_df[feature_cols].values
    y = features_df['Activity Label'].values
    subjects = features_df['Subject-id'].values
    
    # Reshape para CNN: (samples, timesteps=1, features)
    # Cada ventana de 5s es una muestra individual
    X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])

    return X_reshaped, y, subjects #, feature_cols

def get_features_split(data_features_combined: pd.DataFrame | pl.DataFrame):
    X, y, _ = prepare_features_for_cnn_lstm_direct(data_features_combined)

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    print(f"Número de clases: {num_classes}")
    print(f"Clases: {label_encoder.classes_}")

    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_scaled)
    X_scaled = X_scaled.reshape(X.shape)

    # División train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    return (X_train, X_test, X_val), (y_train, y_test, y_val), label_encoder