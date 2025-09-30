import pandas as pd
import polars as pl
import numpy as np
import warnings

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