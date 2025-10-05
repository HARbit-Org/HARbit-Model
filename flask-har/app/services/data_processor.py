from app.logic.window_features_multimodal import create_multimodal_windows_with_features
from app.logic.multimodal import create_multimodal_windows_robust
from app.utils.common import normalize_columns, convert_timestamp
from typing import Any, Dict, List
from datetime import datetime
import os

import tensorflow as tf
import polars as pl
import numpy as np
import joblib

# Configurar TensorFlow para evitar warnings adicionales
tf.get_logger().setLevel('ERROR')

# Cargar modelo y encoder una sola vez al importar el módulo
MODEL_PATH = r"F:\UPC\Tesis\HARbit-Model\src\cnn_temporal_20_epochs_93\saved_model"
ENCODER_PATH = r'F:\UPC\Tesis\HARbit-Model\src\cnn_temporal_20_epochs_93\label_encoder.joblib'

try:
    loaded_model = tf.saved_model.load(MODEL_PATH)
    infer = loaded_model.signatures["serving_default"]
    label_encoder = joblib.load(ENCODER_PATH)
    print("Modelo y encoder cargados exitosamente")
except Exception as e:
    print(f"Error cargando modelo o encoder: {e}")
    loaded_model = None
    infer = None
    label_encoder = None

def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Validar que el modelo esté cargado
        if infer is None or label_encoder is None:
            raise Exception("Modelo o encoder no están disponibles")
            
        # Validar que los datos contengan las claves esperadas
        if 'gyro' not in data or 'accel' not in data:
            raise ValueError("Los datos deben contener 'gyro' y 'accel'")
        
        # Validar que sean listas
        if not isinstance(data['gyro'], list) or not isinstance(data['accel'], list):
            raise ValueError("'gyro' y 'accel' deben ser listas")
        
        # Validar que las listas no estén vacías
        if len(data['gyro']) == 0 or len(data['accel']) == 0:
            raise ValueError("Los datos de 'gyro' y 'accel' no pueden estar vacíos")

        # Crear DataFrames de Polars
        accel_temp = pl.DataFrame(data['accel'])
        # gyro_temp = pl.DataFrame(data['gyro'])

        # Agregar columnas requeridas
        accel_temp = accel_temp.with_columns([
            pl.lit('_').alias('Usuario'),
            pl.lit('-').alias('gt')
        ])

        # Normalizar columnas
        df_accel = normalize_columns(
            accel_temp,
            user_col_name="Usuario", 
            timestamp_col_name="timestamp", 
            label_col_name="gt", 
            x_col_name="x", 
            y_col_name="y", 
            z_col_name="z"
        )
        
        # Convertir timestamps
        df_accel = convert_timestamp(df_accel)
        # df_gyro = convert_timestamp(df_gyro)

        # Crear ventanas con características
        X_all, _, subjects_all, metadata_all = create_multimodal_windows_robust(
            df_accel = df_accel,
            window_seconds=5,
            overlap_percent=50,
            sampling_rate=20,
            target_timesteps=100,
            min_data_threshold=0.8,  # 80% mínimo de datos
            max_gap_seconds=1.0      # Máximo 1 segundo de gap
        )

        # Validar que se generaron ventanas
        if len(X_all) == 0:
            raise ValueError("No se pudieron generar ventanas de datos válidas")

        # Convertir a tensores TensorFlow con tipo de dato específico
        X_tensor = tf.constant(X_all, dtype=tf.float32)

        # Realizar predicción
        y_pred = infer(X_tensor)

        # # Obtener probabilidades y convertir a clases
        y_pred_probs = list(y_pred.values())[0].numpy()
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_pred_classes = label_encoder.inverse_transform(y_pred_classes)

        # # Preparar respuesta
        processed_data = []
        current_time = datetime.now().isoformat()

        for i in range(len(X_all)):
            processed_data.append({
                'window_start': metadata_all.loc[i, 'window_start'].isoformat(),
                'window_end': metadata_all.loc[i, 'window_end'].isoformat(),
                'predicted_activity': str(y_pred_classes[i]),  # Asegurar que sea string
                'model_version': 'CNNTEMP20ACCEL93',
                'created_at': current_time
            })

        return processed_data
        
    except Exception as e:
        raise Exception(f"Error procesando datos: {str(e)}")