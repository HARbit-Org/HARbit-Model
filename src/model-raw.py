from io import StringIO

import matplotlib.pyplot as plt
from scipy.io import arff
import seaborn as sns
from loguru import logger
import yaml

from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
import sys
import os
import json
# from functions.multimodal import create_multimodal_windows_robust
from functions.multimodal_dom_temporal_frequence import create_multimodal_windows_with_features
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import gc, time
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

# MODEL
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for g in gpus: tf.config.experimental.set_memory_growth(g, True)
# tf.keras.mixed_precision.set_global_policy("float32")
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(False)  # belt-and-suspenders: keep XLA off
print("GPUs:", gpus)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler,
    label_binarize
)
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve
)

# Save variables for model
import joblib

# PERSONAL FUNCTIONS
from utils import *
from models.main import *
# from functions.windows import create_feature_windows # creación de ventanas e ingenieria de características

ACT = Path(__file__).resolve().parents[0] / "config" / "activities.yaml"
with open(ACT, "r") as f:
    config = yaml.safe_load(f)["config"]

activities_ = config['labels']
cluster_ = config['clusters']

BATCH_SIZE = 256

def create_raw_windows_250_timesteps_robust(df, window_seconds=5, overlap_percent=50, 
                                           sampling_rate=20, target_timesteps=250,
                                           min_data_threshold=0.5, max_gap_seconds=1.0):
    """
    Versión ROBUSTA: Crea ventanas basadas en TIEMPO REAL con validación mejorada
    
    Args:
        df: DataFrame con datos de sensores (Polars o Pandas)
        window_seconds: Duración de la ventana en segundos (default: 5)
        overlap_percent: Porcentaje de solapamiento (default: 50)
        sampling_rate: Frecuencia de muestreo en Hz (default: 20)
        target_timesteps: Número objetivo de timesteps por ventana (default: 250)
        min_data_threshold: Umbral mínimo de datos válidos (0.5 = 50%)
        max_gap_seconds: Máximo gap permitido en segundos (1.0s)
        
    Returns:
        X: Array con forma (n_windows, 250, 3) - datos de ventanas
        y: Array con etiquetas de actividad
        subjects: Array con IDs de usuario
        metadata: DataFrame con información de las ventanas
    """
    
    print(f"🔧 Configuración de ventanas RAW ROBUSTA:")
    print(f"  Duración: {window_seconds}s")
    print(f"  Timesteps objetivo: {target_timesteps}")
    print(f"  Frecuencia de muestreo: {sampling_rate}Hz")
    print(f"  Solapamiento: {overlap_percent}%")
    print(f"  Umbral mínimo de datos: {min_data_threshold*100:.1f}%")
    print(f"  Máximo gap permitido: {max_gap_seconds}s")
    
    # Convertir a pandas si es necesario
    if hasattr(df, 'to_pandas'):
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy()
    
    # Asegurar que Timestamp es datetime
    if df_pd['Timestamp'].dtype == 'object':
        df_pd['Timestamp'] = pd.to_datetime(df_pd['Timestamp'])
    elif df_pd['Timestamp'].dtype == 'int64':
        df_pd['Timestamp'] = pd.to_datetime(df_pd['Timestamp'])
    
    # Calcular parámetros de tiempo
    window_duration_ns = int(window_seconds * 1e9)
    step_duration_ns = int(window_duration_ns * (100 - overlap_percent) / 100)
    
    print(f"  Duración de ventana: {window_seconds}s")
    print(f"  Paso entre ventanas: {step_duration_ns / 1e9:.2f}s")
    
    # Listas para almacenar resultados
    X_windows = []
    y_labels = []
    subjects_list = []
    metadata_list = []
    
    total_windows_attempted = 0
    total_windows_created = 0
    
    # Procesar por usuario y actividad
    for (user_id, activity), group in df_pd.groupby(['Subject-id', 'Activity Label']):
        
        # Ordenar por timestamp y limpiar datos
        group = group.sort_values('Timestamp').reset_index(drop=True)
        group = group.dropna(subset=['X', 'Y', 'Z', 'Timestamp'])
        
        if len(group) < window_seconds * sampling_rate:
            print(f"⚠️ Usuario {user_id}, Actividad {activity}: Muy pocos datos ({len(group)} muestras)")
            continue
        
        # Convertir timestamps a nanosegundos
        if group['Timestamp'].dtype.name.startswith('datetime'):
            timestamps_ns = group['Timestamp'].astype('int64')
        else:
            timestamps_ns = group['Timestamp'].values
        
        print(f"👤 Usuario {user_id}, Actividad {activity}: {len(group)} muestras")
        
        # Obtener rango temporal
        start_time_ns = timestamps_ns.min()
        end_time_ns = timestamps_ns.max()
        total_duration_s = (end_time_ns - start_time_ns) / 1e9
        
        print(f"   Duración total: {total_duration_s:.1f}s")
        
        # Detectar y reportar gaps grandes
        time_diffs = np.diff(timestamps_ns) / 1e9  # Convertir a segundos
        large_gaps = time_diffs > max_gap_seconds
        if np.any(large_gaps):
            n_gaps = np.sum(large_gaps)
            max_gap = np.max(time_diffs)
            print(f"   ⚠️ Detectados {n_gaps} gaps > {max_gap_seconds}s (máximo: {max_gap:.1f}s)")
        
        # Crear ventanas deslizantes
        window_count = 0
        current_start_ns = start_time_ns
        
        while current_start_ns + window_duration_ns <= end_time_ns:
            total_windows_attempted += 1
            current_end_ns = current_start_ns + window_duration_ns
            
            # Filtrar datos en esta ventana temporal
            window_mask = (
                (timestamps_ns >= current_start_ns) & 
                (timestamps_ns < current_end_ns)
            )
            window_data_df = group[window_mask]
            
            # Validación de ventana
            is_valid, validation_info = validate_window_data(
                window_data_df, 
                window_seconds, 
                sampling_rate, 
                min_data_threshold,
                max_gap_seconds
            )
            
            if is_valid:
                # Extraer datos de sensores
                sensor_data = window_data_df[['X', 'Y', 'Z']].values
                window_timestamps = window_data_df['Timestamp'].values
                
                try:
                    # Redimensionar/interpolar a target_timesteps
                    resampled_window = resample_window_robust(
                        sensor_data, window_timestamps, target_timesteps, window_seconds
                    )
                    
                    # Verificar calidad final
                    if is_window_quality_good(resampled_window):
                        # Guardar datos
                        X_windows.append(resampled_window)
                        y_labels.append(activity)
                        subjects_list.append(user_id)
                        
                        # Metadata extendida
                        metadata_list.append({
                            'Subject-id': user_id,
                            'Activity Label': activity,
                            'window_start': pd.to_datetime(current_start_ns),
                            'window_end': pd.to_datetime(current_end_ns),
                            'original_samples': len(window_data_df),
                            'resampled_timesteps': target_timesteps,
                            'window_idx': window_count,
                            'actual_duration_s': window_seconds,
                            'data_coverage': validation_info['data_coverage'],
                            'max_gap_s': validation_info['max_gap'],
                            'sampling_rate_actual': validation_info['actual_rate']
                        })
                        
                        window_count += 1
                        total_windows_created += 1
                    else:
                        print(f"   ❌ Ventana {window_count}: Calidad de datos insuficiente después de interpolación")
                
                except Exception as e:
                    print(f"   ❌ Ventana {window_count}: Error en interpolación - {str(e)}")
            
            else:
                # No mostrar warning para cada ventana inválida, solo resumen
                pass
            
            # Mover al siguiente inicio de ventana
            current_start_ns += step_duration_ns
        
        print(f"  ✅ Creadas {window_count} ventanas válidas")
    
    # Resumen final
    print(f"\n📊 RESUMEN DE VALIDACIÓN:")
    print(f"  Ventanas intentadas: {total_windows_attempted}")
    print(f"  Ventanas creadas: {total_windows_created}")
    print(f"  Tasa de éxito: {(total_windows_created/total_windows_attempted)*100:.1f}%")
    
    # Convertir a arrays numpy
    if len(X_windows) > 0:
        X = np.array(X_windows)
        y = np.array(y_labels)
        subjects = np.array(subjects_list)
        metadata_df = pd.DataFrame(metadata_list)
        
        print(f"\n📊 RESULTADO FINAL (ROBUSTO):")
        print(f"  Forma de X: {X.shape}")
        print(f"  Forma de y: {y.shape}")
        print(f"  Total ventanas: {len(X)}")
        print(f"  Usuarios únicos: {len(np.unique(subjects))}")
        print(f"  Actividades únicas: {sorted(np.unique(y))}")
        
        return X, y, subjects, metadata_df
    
    else:
        print("❌ No se crearon ventanas válidas")
        return None, None, None, None


def validate_window_data(window_data_df, window_seconds, sampling_rate, 
                        min_data_threshold, max_gap_seconds):
    """
    Valida si una ventana de datos es aceptable
    
    Returns:
        bool: True si la ventana es válida
        dict: Información de validación
    """
    if len(window_data_df) == 0:
        return False, {'reason': 'empty', 'data_coverage': 0, 'max_gap': float('inf'), 'actual_rate': 0}
    
    # Calcular cobertura de datos esperada
    expected_samples = window_seconds * sampling_rate
    actual_samples = len(window_data_df)
    data_coverage = actual_samples / expected_samples
    
    # Si hay muy pocos datos
    if data_coverage < min_data_threshold:
        return False, {
            'reason': 'insufficient_data', 
            'data_coverage': data_coverage,
            'max_gap': float('inf'),
            'actual_rate': 0
        }
    
    # Calcular gaps en los datos
    if len(window_data_df) > 1:
        timestamps = pd.to_datetime(window_data_df['Timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds().fillna(0)
        max_gap = time_diffs.max()
        actual_rate = len(window_data_df) / (timestamps.max() - timestamps.min()).total_seconds()
    else:
        max_gap = 0
        actual_rate = sampling_rate
    
    # Si hay gaps muy grandes
    if max_gap > max_gap_seconds:
        return False, {
            'reason': 'large_gap', 
            'data_coverage': data_coverage,
            'max_gap': max_gap,
            'actual_rate': actual_rate
        }
    
    # Verificar que no hay valores NaN o infinitos en los sensores
    sensor_data = window_data_df[['X', 'Y', 'Z']].values
    if np.any(np.isnan(sensor_data)) or np.any(np.isinf(sensor_data)):
        return False, {
            'reason': 'invalid_values',
            'data_coverage': data_coverage,
            'max_gap': max_gap,
            'actual_rate': actual_rate
        }
    
    return True, {
        'reason': 'valid',
        'data_coverage': data_coverage,
        'max_gap': max_gap,
        'actual_rate': actual_rate
    }


def resample_window_robust(sensor_data, timestamps, target_timesteps, window_seconds):
    """
    Versión robusta de remuestreo con múltiples estrategias
    """
    from scipy.interpolate import interp1d
    from scipy import signal
    
    if len(sensor_data) == 0:
        return np.zeros((target_timesteps, 3))
    
    original_timesteps = len(sensor_data)
    
    if original_timesteps == target_timesteps:
        return sensor_data.copy()
    
    if original_timesteps == 1:
        return np.tile(sensor_data[0], (target_timesteps, 1))
    
    try:
        # Estrategia 1: Interpolación temporal precisa
        if hasattr(timestamps[0], 'timestamp'):
            time_seconds = np.array([t.timestamp() for t in timestamps])
        elif isinstance(timestamps[0], pd.Timestamp):
            time_seconds = np.array([t.timestamp() for t in timestamps])
        else:
            time_seconds = timestamps.astype('int64') / 1e9
        
        # Normalizar tiempos
        time_min = time_seconds.min()
        time_max = time_seconds.max()
        
        if time_max > time_min:
            relative_times = (time_seconds - time_min) / (time_max - time_min)
        else:
            relative_times = np.linspace(0, 1, len(time_seconds))
        
        # Crear tiempos objetivo uniformes
        target_times = np.linspace(0, 1, target_timesteps)
        
        # Interpolar cada eje
        resampled_data = np.zeros((target_timesteps, 3))
        
        for axis in range(3):
            try:
                # Estrategia de interpolación según la cantidad de datos
                if original_timesteps >= target_timesteps:
                    # Downsample: usar signal.resample para preservar características
                    resampled_axis = signal.resample(sensor_data[:, axis], target_timesteps)
                else:
                    # Upsample: usar interpolación
                    if len(np.unique(relative_times)) > 1:
                        interpolator = interp1d(
                            relative_times, 
                            sensor_data[:, axis],
                            kind='cubic' if original_timesteps >= 4 else 'linear',
                            bounds_error=False,
                            fill_value='extrapolate'
                        )
                        resampled_axis = interpolator(target_times)
                    else:
                        resampled_axis = np.full(target_timesteps, sensor_data[0, axis])
                
                resampled_data[:, axis] = resampled_axis
                
            except Exception as e:
                # Fallback: interpolación lineal simple
                resampled_data[:, axis] = np.interp(
                    target_times, relative_times, sensor_data[:, axis]
                )
        
        return resampled_data
    
    except Exception as e:
        print(f"Error en remuestreo robusto: {str(e)}")
        # Último fallback: replicar la primera muestra
        return np.tile(sensor_data[0], (target_timesteps, 1))


def is_window_quality_good(resampled_window, max_std_threshold=50.0):
    """
    Verifica la calidad final de una ventana remuestreada
    """
    # Verificar NaN o infinitos
    if np.any(np.isnan(resampled_window)) or np.any(np.isinf(resampled_window)):
        return False
    
    # Verificar valores extremos (posibles errores de interpolación)
    if np.any(np.abs(resampled_window) > 1000):  # Ajustar según tus datos
        return False
    
    # Verificar varianza (datos demasiado planos pueden indicar error)
    for axis in range(resampled_window.shape[1]):
        std_axis = np.std(resampled_window[:, axis])
        if std_axis > max_std_threshold:  # Varianza excesiva
            return False
        if std_axis < 0.001:  # Datos demasiado planos
            return False
    
    return True

def loso_cross_validation_raw_windows_robust(df_accel, df_gyro=None, model_architecture_func=None,
                                           window_seconds=5, overlap_percent=50, 
                                           sampling_rate=20, target_timesteps=250,
                                           epochs=30, batch_size=32, verbose=1, 
                                           save_results=True, results_path="loso_raw_windows"):
    """LOSO con validación robusta de ventanas"""
    
    print("🚀 INICIANDO LOSO CROSS-VALIDATION - VENTANAS RAW ROBUSTAS")
    print("=" * 70)
    
    # Usar la función robusta para crear ventanas
    X_all, y_all, subjects_all, metadata_all = create_raw_windows_250_timesteps_robust(
        df=df_accel,
        window_seconds=window_seconds,
        overlap_percent=overlap_percent,
        sampling_rate=sampling_rate,
        target_timesteps=target_timesteps,
        min_data_threshold=0.8,  # 80% mínimo de datos
        max_gap_seconds=1.0      # Máximo 1 segundo de gap
    )

    y_all_mapped = []
    for label in y_all:
        # Primero mapear con activities_
        mapped_activity = activities_[label]
        
        # Luego buscar en qué cluster está
        final_cluster = mapped_activity  # Por defecto, mantener la actividad
        for cluster_name, activities_in_cluster in cluster_.items():
            if mapped_activity in activities_in_cluster:
                final_cluster = cluster_name
                break
        
        y_all_mapped.append(final_cluster)
    
    y_all = np.array(y_all_mapped)
    print("Actividades únicas después del mapeo:", np.unique(y_all))
    
    if X_all is None:
        print("❌ No se pudieron crear ventanas válidas")
        return None
    
    # Obtener usuarios únicos
    users = np.unique(subjects_all)
    n_users = len(users)
    
    print(f"👥 Total de usuarios: {n_users}")
    print(f"📊 Usuarios: {users}")
    print(f"📊 Forma de datos: {X_all.shape}")
    
    # Preparar label encoder global
    from sklearn.preprocessing import LabelEncoder
    global_label_encoder = LabelEncoder()
    y_all_encoded = global_label_encoder.fit_transform(y_all)
    
    # Almacenar resultados
    loso_results = {
        'user': [],
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'confusion_matrix': [],
        'classification_report': [],
        'train_windows': [],
        'test_windows': [],
        'y_true': [],
        'y_pred': []
    }
    
    all_y_true = []
    all_y_pred = []
    
    # Iterar sobre cada usuario (LOSO)
    for i, test_user in enumerate(users):
        print(f"\n🔄 FOLD {i+1}/{n_users}: Usuario de test = {test_user}")
        print("-" * 50)
        
        # División de datos: un usuario para test, resto para train
        train_mask = subjects_all != test_user
        test_mask = subjects_all == test_user

        train_subjects = np.unique(subjects_all[train_mask])

        rng = np.random.default_rng(42 + i)  # i = fold index to vary per fold
        n_val_users = max(1, int(0.10 * len(train_subjects)))
        val_users = rng.choice(train_subjects, size=n_val_users, replace=False)

        val_mask   = train_mask & np.isin(subjects_all, val_users)
        fit_mask   = train_mask & ~np.isin(subjects_all, val_users)
        
        X_fit,  y_fit  = X_all[fit_mask].astype("float32"),  y_all_encoded[fit_mask].astype("int32")
        X_val,  y_val  = X_all[val_mask].astype("float32"),  y_all_encoded[val_mask].astype("int32")
        X_test, y_test = X_all[test_mask].astype("float32"), y_all_encoded[test_mask].astype("int32")

        print(f"📊 Train users: {len(np.unique(subjects_all[fit_mask]))} | "
              f"Val users: {len(np.unique(subjects_all[val_mask]))} | "
              f"Test user: {test_user}")
        
        # Verificar que hay suficientes datos
        if len(X_fit) < 50 or len(X_test) < 10:
            print(f"⚠️ Datos insuficientes para usuario {test_user}, saltando...")
            continue
        
        # Verificar clases en común
        train_classes = set(np.unique(y_fit))
        test_classes = set(np.unique(y_test))
        common_classes = train_classes.intersection(test_classes)
        
        if len(common_classes) < 2:
            print(f"⚠️ Muy pocas clases comunes para usuario {test_user}, saltando...")
            continue
        
        print(f"🎯 Clases comunes: {len(common_classes)}")
        
        try:
            # tf.data pipelines
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = (tf.data.Dataset.from_tensor_slices((X_fit, y_fit))
                        .shuffle(10000)
                        .batch(BATCH_SIZE)
                        .prefetch(AUTOTUNE))
            val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                        .batch(BATCH_SIZE)
                        .prefetch(AUTOTUNE))
            test_ds  = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
                        .batch(BATCH_SIZE)
                        .prefetch(AUTOTUNE))
            # Crear y compilar modelo
            input_shape = (target_timesteps, X_fit.shape[2])   # ✅ use X_fit
            num_classes = len(global_label_encoder.classes_)

            model = (create_cnn_lstm_model(input_shape=input_shape, num_classes=num_classes)
                    if model_architecture_func is None
                    else model_architecture_func(input_shape=input_shape, num_classes=num_classes))
            
            # Entrenar modelo
            print("🏋️ Entrenando modelo...")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1 if verbose else 0
            )
            
            # Evaluar modelo
            print("🔍 Evaluando modelo...")
            y_pred = model.predict(test_ds, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
            accuracy = accuracy_score(y_test, y_pred_classes)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred_classes, average='macro', zero_division=0
            )
            report = classification_report(
                y_test, y_pred_classes,
                target_names=global_label_encoder.classes_,
                output_dict=True,
                zero_division=0
            )
            cm = confusion_matrix(y_test, y_pred_classes)
            
            # Guardar resultados
            loso_results['user'].append(test_user)
            loso_results['accuracy'].append(accuracy)
            loso_results['precision_macro'].append(precision)
            loso_results['recall_macro'].append(recall)
            loso_results['f1_macro'].append(f1)
            loso_results['confusion_matrix'].append(cm)
            loso_results['classification_report'].append(report)
            loso_results['train_windows'].append(len(X_fit))   # ✅
            loso_results['test_windows'].append(len(X_test))
            loso_results['y_true'].append(y_test)
            loso_results['y_pred'].append(y_pred_classes)
            
            # Acumular para métricas globales    
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred_classes)
            
            print(f"✅ Usuario {test_user}: Accuracy = {accuracy:.4f}")

            del train_ds, val_ds, test_ds, X_fit, y_fit, X_val, y_val, X_test, y_test
            tf.keras.backend.clear_session()
            gc.collect()
            time.sleep(0.2)
            
        except Exception as e:
            print(f"❌ Error con usuario {test_user}: {str(e)}")
            continue
    
    # Calcular métricas agregadas
    print(f"\n📊 RESULTADOS LOSO RAW WINDOWS")
    print("=" * 70)
    
    if len(loso_results['accuracy']) > 0:
        mean_accuracy = np.mean(loso_results['accuracy'])
        std_accuracy = np.std(loso_results['accuracy'])
        mean_precision = np.mean(loso_results['precision_macro'])
        mean_recall = np.mean(loso_results['recall_macro'])
        mean_f1 = np.mean(loso_results['f1_macro'])
        
        print(f"🎯 Accuracy promedio: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"🎯 Precision promedio: {mean_precision:.4f}")
        print(f"🎯 Recall promedio: {mean_recall:.4f}")
        print(f"🎯 F1-Score promedio: {mean_f1:.4f}")
        print(f"📊 Usuarios evaluados: {len(loso_results['accuracy'])}/{n_users}")
        
        # Guardar resultados
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results_df = pd.DataFrame({
                'user': loso_results['user'],
                'accuracy': loso_results['accuracy'],
                'precision_macro': loso_results['precision_macro'],
                'recall_macro': loso_results['recall_macro'],
                'f1_macro': loso_results['f1_macro'],
                'train_windows': loso_results['train_windows'],
                'test_windows': loso_results['test_windows']
            })
            
            results_df.to_csv(f"{results_path}_summary_{timestamp}.csv", index=False)
            joblib.dump(loso_results, f"{results_path}_complete_{timestamp}.joblib")
            
            print(f"💾 Resultados guardados: {results_path}_*_{timestamp}.*")
        
        return {
            'summary': results_df,
            'model': model,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'detailed_results': loso_results,
            'global_label_encoder': global_label_encoder
        }
    
    else:
        print("❌ No se pudieron evaluar usuarios")
        return None

# Cargar datos
SRC_DIR   = Path(__file__).resolve().parent
path_base = SRC_DIR / "data" / "wisdm-dataset" / "raw" / "watch"
sensor_data = load_sensors_separately(path_base)
df_gyro = sensor_data['gyro']
# df_gyro = df_gyro.rename({'X': 'X_gyro', 'Y': 'Y_gyro', 'Z': 'Z_gyro'})

df_accel = sensor_data['accel']
# df_accel = df_accel.rename({'X': 'X_accel', 'Y': 'Y_accel', 'Z': 'Z_accel'})

print(f"Giroscopio: {len(df_gyro)} muestras")
print(f"Acelerómetro: {len(df_accel)} muestras")

# # Luego ejecuta el LOSO corregido
# print("\n🚀 Ejecutando LOSO con ventanas RAW CORREGIDAS...")

# # Ejecutar LOSO con la función corregida
# loso_results = loso_cross_validation_raw_windows_robust(
#     df_accel=df_accel,
#     df_gyro=None,
#     model_architecture_func=create_cnn_lstm_model,
#     window_seconds=5,
#     overlap_percent=50,
#     sampling_rate=20,
#     target_timesteps=100,
#     epochs=20,
#     batch_size=BATCH_SIZE,
#     verbose=1,
#     save_results=True,
#     results_path="loso_raw_robust"
# )

# # Mostrar resultados
# if loso_results:
#     print(f"✅ LOSO RAW completado!")
#     print(f"📊 Accuracy promedio: {loso_results['mean_accuracy']:.4f}")
#     print(f"📊 Std Accuracy: {loso_results['std_accuracy']:.4f}")
# else:
#     print("❌ LOSO falló - revisa los datos")

def train_final_model_on_full_data(
    df_accel,
    model_architecture_func,
    window_seconds=5,
    overlap_percent=50,
    sampling_rate=20,
    target_timesteps=100,
    epochs=20,
    batch_size=256,
    results_dir="final_model"
):
    """
    Fit a final model on (almost) all subjects using the same pipeline as LOSO.
    Keeps ~10% of subjects for validation (subject-disjoint) to drive callbacks.
    Saves model + encoder + run metadata.
    """
    print("🚀 Building windows for FINAL training…")
    # X_all, y_all, subjects_all, metadata_all = create_multimodal_windows_robust(
    X_all, X_features, y_all, subjects_all, metadata_all = create_multimodal_windows_with_features(
        df_accel=df_accel,
        df_gyro=df_gyro,
        window_seconds=window_seconds,
        overlap_percent=overlap_percent,
        sampling_rate=sampling_rate,
        target_timesteps=target_timesteps,
        min_data_threshold=0.8,
        max_gap_seconds=1.0
    )
    if X_all is None or len(X_all) == 0:
        raise RuntimeError("No valid windows were created for final training.")

    # Map raw label -> activity -> cluster (same as your LOSO code)
    y_all_mapped = []
    for label in y_all:
        mapped_activity = activities_[label]
        final_cluster = mapped_activity
        for cluster_name, activities_in_cluster in cluster_.items():
            if mapped_activity in activities_in_cluster:
                final_cluster = cluster_name
                break
        y_all_mapped.append(final_cluster)
    y_all = np.array(y_all_mapped)
    print("✅ Unique activities after mapping:", np.unique(y_all))

    # Encode labels (global)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_all_encoded = le.fit_transform(y_all).astype("int32")

    # Subject-disjoint small validation split (~10%)
    users = np.unique(subjects_all)
    rng = np.random.default_rng(2025)
    n_val_users = max(1, int(0.10 * len(users)))
    val_users = rng.choice(users, size=n_val_users, replace=False)

    val_mask = np.isin(subjects_all, val_users)
    train_mask = ~val_mask

    X_train = X_all[train_mask].astype("float32")
    X_val   = X_all[val_mask].astype("float32")
    X_train_feat = X_features[train_mask].astype("float32")
    X_val_feat   = X_features[val_mask].astype("float32")

    y_train = y_all_encoded[train_mask]
    y_val   = y_all_encoded[val_mask]
    print(f"👥 Train users: {len(np.unique(subjects_all[train_mask]))} | "
          f"Val users: {len(np.unique(subjects_all[val_mask]))}")
    print(f"📦 Train windows: {len(X_train)} | Val windows: {len(X_val)}")

    # tf.data input pipelines
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (tf.data.Dataset.from_tensor_slices(((X_train, X_train_feat), y_train))
                .shuffle(min(10000, len(X_train)))
                .batch(batch_size)
                .prefetch(AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices(((X_val, X_val_feat), y_val))
              .batch(batch_size)
              .prefetch(AUTOTUNE))

    # Build model with the same hyperparams/arch you used in LOSO
    input_shape_raw  = (target_timesteps, X_train.shape[2])
    input_shape_feat = (X_train_feat.shape[1],)
    num_classes = len(le.classes_)

    model = model_architecture_func(
        input_shape_raw=input_shape_raw,
        input_shape_feat=input_shape_feat,
        num_classes=num_classes
    )
    
    # If training with mixed_float16, ensure float32 output for stable loss/metrics
    try:
        if model.output.dtype != tf.float32:
            out = tf.keras.layers.Activation("linear", dtype="float32", name="float32_head")(model.output)
            model = tf.keras.Model(model.input, out, name=model.name)
    except Exception:
        pass

    print("🏋️ Training final model…")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,  # comes from your `models.main`
        verbose=1
    )

    # Create run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(results_dir) / f"run_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Save model (both formats)
    model.save(outdir / "saved_model")
    model.save(outdir / "model.keras")
    model.save_weights(outdir / "weights.h5")

    # Save label encoder and classes
    joblib.dump(le, outdir / "label_encoder.joblib")
    (outdir / "classes.json").write_text(json.dumps(le.classes_.tolist(), ensure_ascii=False, indent=2))

    # Save a tiny run summary
    summary = {
        "timestamp": timestamp,
        "input_shape": list(input_shape),
        "num_classes": int(num_classes),
        "train_windows": int(len(X_train)),
        "val_windows": int(len(X_val)),
        "val_users": list(map(lambda x: x if isinstance(x, (int, float)) else str(x), np.unique(subjects_all[val_mask]))),
        "window_seconds": window_seconds,
        "overlap_percent": overlap_percent,
        "sampling_rate": sampling_rate,
        "target_timesteps": target_timesteps,
        "epochs": epochs,
        "batch_size": batch_size,
        "activities_after_mapping": list(map(str, np.unique(y_all))),
        "history_keys": list(history.history.keys()),
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"💾 Saved artifacts to: {outdir}")
    return {
        "model": model,
        "label_encoder": le,
        "artifacts_dir": str(outdir),
        "val_metrics": {k: history.history[k][-1] for k in history.history if k.startswith("val_")}
    }

final_run = train_final_model_on_full_data(
    df_accel=df_accel,
    model_architecture_func=create_cnn_lstm_model,  # or your CNN-only if you prefer
    window_seconds=5,
    overlap_percent=50,
    sampling_rate=20,
    target_timesteps=100,      # <- match what you used last
    epochs=20,
    batch_size=256,
    results_dir="final_model"
)

print("Artifacts dir:", final_run["artifacts_dir"])