import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_feature_sequences(features_df, group_size=5, step_size=1):
    """
    Convierte features de ventanas en secuencias para CNN-LSTM
    Args:
        features_df: DataFrame con columnas de características + Subject-id + Activity Label
        group_size: número de ventanas consecutivas por secuencia
        step_size: desplazamiento del sliding window entre secuencias
    Returns:
        X_seq: np.array con forma (N, group_size, num_features)
        y_seq: etiquetas correspondientes
        subjects_seq: id del sujeto para cada secuencia
    """
    # Convertir a pandas si es Polars
    if hasattr(features_df, 'to_pandas'):
        features_df = features_df.to_pandas()

    metadata_cols = ['Subject-id', 'Activity Label', 'window_start', 'window_end', 'sample_count']
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]
    num_features = len(feature_cols)

    sequences = []
    labels = []
    subjects = []

    # Agrupar por sujeto y actividad para no mezclar
    grouped = features_df.groupby(['Subject-id', 'Activity Label'])
    for (subject, activity), group in grouped:
        group = group.sort_values('window_start').reset_index(drop=True)
        data = group[feature_cols].values

        # Generar secuencias deslizantes
        for i in range(0, len(group) - group_size + 1, step_size):
            seq = data[i:i + group_size]
            sequences.append(seq)
            labels.append(activity)  # puedes usar la última ventana o la moda
            subjects.append(subject)

    X_seq = np.array(sequences)  # (N, group_size, num_features)
    y_seq = np.array(labels)
    subjects_seq = np.array(subjects)

    return X_seq, y_seq, subjects_seq

def prepare_features_for_cnn_lstm_sequences(features_df, group_size=5, step_size=1):
    """
    Prepara features en secuencias para CNN-LSTM con dependencias temporales
    """
    X, y, subjects = create_feature_sequences(features_df, group_size, step_size)

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    print(f"✅ Secuencias creadas: {X.shape}")
    print(f"  Num features: {X.shape[2]}")
    print(f"  Clases: {label_encoder.classes_}")

    # Normalizar en feature-level
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)

    return X_scaled, y_encoded, subjects, label_encoder
