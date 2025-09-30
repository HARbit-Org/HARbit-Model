import pandas as pd
import polars as pl
import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy import signal


def prepare_sensor_dataframe(df, sensor_type):
    """Prepara y limpia DataFrame de sensor"""
    if df is None:
        return None
    
    # Convertir a pandas si es necesario
    if hasattr(df, 'to_pandas'):
        df_clean = df.to_pandas()
    else:
        df_clean = df.copy()
    
    # Asegurar formato correcto de timestamp
    if df_clean['Timestamp'].dtype == 'object':
        df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
    elif df_clean['Timestamp'].dtype == 'int64':
        df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
    
    # Limpiar datos
    df_clean = df_clean.dropna(subset=['X', 'Y', 'Z', 'Timestamp'])
    df_clean = df_clean.sort_values('Timestamp').reset_index(drop=True)
    
    print(f"  {sensor_type.upper()} preparado: {len(df_clean)} muestras")
    
    return df_clean


def synchronize_multimodal_data(df_accel, df_gyro, sync_tolerance_ms):
    """
    Sincroniza datos de acelerómetro y giroscopio con tolerancia temporal optimizada
    """
    
    # Convertir timestamps a nanosegundos para precisión
    accel_times_ns = df_accel['Timestamp'].astype('int64')
    gyro_times_ns = df_gyro['Timestamp'].astype('int64')
    
    print(f"  📊 Datos originales:")
    print(f"    Acelerómetro: {len(df_accel):,} muestras")
    print(f"    Giroscopio: {len(df_gyro):,} muestras")
    
    # Encontrar rango temporal común
    common_start = max(accel_times_ns.min(), gyro_times_ns.min())
    common_end = min(accel_times_ns.max(), gyro_times_ns.max())
    
    print(f"  ⏰ Rango temporal común: {(common_end - common_start) / 1e9:.1f}s")
    
    # Filtrar datos al rango común
    accel_mask = (accel_times_ns >= common_start) & (accel_times_ns <= common_end)
    gyro_mask = (gyro_times_ns >= common_start) & (gyro_times_ns <= common_end)
    
    df_accel_sync = df_accel[accel_mask].copy()
    df_gyro_sync = df_gyro[gyro_mask].copy()
    
    print(f"  📊 Datos sincronizados:")
    print(f"    Acelerómetro: {len(df_accel_sync):,} muestras")
    print(f"    Giroscopio: {len(df_gyro_sync):,} muestras")
    
    # Análisis rápido de calidad de sincronización
    # analyze_sync_quality_fast(df_accel_sync, df_gyro_sync, sync_tolerance_ms)
    
    return df_accel_sync, df_gyro_sync


def analyze_sync_quality_fast(df_accel, df_gyro, tolerance_ms, sample_size=10000):
    """Análisis rápido de calidad de sincronización"""
    
    print(f"  🔍 Análisis de sincronización (muestra de {sample_size:,})...")
    
    accel_times = df_accel['Timestamp'].values.astype('int64')
    gyro_times = df_gyro['Timestamp'].values.astype('int64')
    
    # Muestreo estratificado
    total_accel = len(accel_times)
    if total_accel > sample_size:
        indices = np.linspace(0, total_accel-1, sample_size, dtype=int)
        accel_sample = accel_times[indices]
    else:
        accel_sample = accel_times
    
    # Análisis vectorizado en muestra
    tolerance_ns = tolerance_ms * 1e6
    diffs_ns = np.abs(accel_sample[:, None] - gyro_times[None, :])
    min_diffs_ns = np.min(diffs_ns, axis=1)
    min_diffs_ms = min_diffs_ns / 1e6
    
    valid_diffs = min_diffs_ms[min_diffs_ms <= tolerance_ms]
    matched_pairs = len(valid_diffs)
    
    if matched_pairs > 0:
        sync_rate = (matched_pairs / len(accel_sample)) * 100
        avg_diff = float(np.mean(valid_diffs))
        
        print(f"    ✅ Sincronización: {sync_rate:.1f}% (diff. promedio: {avg_diff:.1f}ms)")
    else:
        print(f"    ⚠️ Baja sincronización: <{tolerance_ms}ms")


def process_multimodal_window_robust(window_accel, window_gyro, target_timesteps,
                                   window_seconds, min_data_threshold, max_gap_seconds,
                                   start_time_ns, end_time_ns, sampling_rate):
    """Procesa una ventana multimodal individual"""
    
    # Validar acelerómetro
    accel_valid, accel_info = validate_window_data(
        window_accel, window_seconds, sampling_rate, min_data_threshold, max_gap_seconds
    )
    
    if not accel_valid:
        return {
            'is_valid': False,
            'reason': f"accel_{accel_info['reason']}",
            'start_time': pd.to_datetime(start_time_ns),
            'end_time': pd.to_datetime(end_time_ns),
            'accel_samples': len(window_accel),
            'gyro_samples': len(window_gyro) if window_gyro is not None else 0,
            'sync_quality': 0.0,
            'data_coverage': accel_info['data_coverage'],
            'max_gap': accel_info['max_gap']
        }
    
    # Procesar datos de acelerómetro
    accel_data = window_accel[['X', 'Y', 'Z']].values
    accel_timestamps = window_accel['Timestamp'].values
    
    try:
        accel_resampled = resample_window_robust(
            accel_data, accel_timestamps, target_timesteps, window_seconds
        )
    except Exception as e:
        return {
            'is_valid': False,
            'reason': f"accel_resample_error: {str(e)[:50]}",
            'start_time': pd.to_datetime(start_time_ns),
            'end_time': pd.to_datetime(end_time_ns),
            'accel_samples': len(window_accel),
            'gyro_samples': len(window_gyro) if window_gyro is not None else 0,
            'sync_quality': 0.0,
            'data_coverage': accel_info['data_coverage'],
            'max_gap': accel_info['max_gap']
        }
    
    # Procesar giroscopio si está disponible
    if window_gyro is not None and len(window_gyro) > 0:
        gyro_valid, gyro_info = validate_window_data(
            window_gyro, window_seconds, sampling_rate, min_data_threshold, max_gap_seconds
        )
        
        if gyro_valid:
            gyro_data = window_gyro[['X', 'Y', 'Z']].values
            gyro_timestamps = window_gyro['Timestamp'].values
            
            try:
                gyro_resampled = resample_window_robust(
                    gyro_data, gyro_timestamps, target_timesteps, window_seconds
                )
                
                # Verificar calidad final multimodal
                if (is_window_quality_good(accel_resampled) and 
                    is_window_quality_good(gyro_resampled)):
                    
                    # Concatenar datos multimodales
                    combined_data = np.concatenate([accel_resampled, gyro_resampled], axis=1)
                    
                    return {
                        'is_valid': True,
                        'sensor_data': combined_data,
                        'start_time': pd.to_datetime(start_time_ns),
                        'end_time': pd.to_datetime(end_time_ns),
                        'accel_samples': len(window_accel),
                        'gyro_samples': len(window_gyro),
                        'sync_quality': 1.0,  # Simplificado para eficiencia
                        'data_coverage': min(accel_info['data_coverage'], gyro_info['data_coverage']),
                        'max_gap': max(accel_info['max_gap'], gyro_info['max_gap'])
                    }
                
            except Exception as e:
                # Fallback: solo acelerómetro si falla giroscopio
                pass
    
    # Solo acelerómetro (fallback o modo monomodal)
    if is_window_quality_good(accel_resampled):
        return {
            'is_valid': True,
            'sensor_data': accel_resampled,
            'start_time': pd.to_datetime(start_time_ns),
            'end_time': pd.to_datetime(end_time_ns),
            'accel_samples': len(window_accel),
            'gyro_samples': len(window_gyro) if window_gyro is not None else 0,
            'sync_quality': 0.5,  # Parcial porque solo tiene un sensor
            'data_coverage': accel_info['data_coverage'],
            'max_gap': accel_info['max_gap']
        }
    else:
        return {
            'is_valid': False,
            'reason': 'poor_quality_after_resampling',
            'start_time': pd.to_datetime(start_time_ns),
            'end_time': pd.to_datetime(end_time_ns),
            'accel_samples': len(window_accel),
            'gyro_samples': len(window_gyro) if window_gyro is not None else 0,
            'sync_quality': 0.0,
            'data_coverage': accel_info['data_coverage'],
            'max_gap': accel_info['max_gap']
        }


# FUNCIÓN DE CONVENIENCIA PARA USO DIRECTO
def create_raw_windows_multimodal(df_accel, df_gyro=None, window_seconds=5, 
                                target_timesteps=100, **kwargs):
    """
    Función simplificada para crear ventanas multimodales
    """
    return create_multimodal_windows_robust(
        df_accel=df_accel,
        df_gyro=df_gyro,
        window_seconds=window_seconds,
        target_timesteps=target_timesteps,
        **kwargs
    )


# Mantener las funciones auxiliares originales
def validate_window_data(window_data_df, window_seconds, sampling_rate, 
                        min_data_threshold, max_gap_seconds):
    """Valida si una ventana de datos es aceptable"""
    if len(window_data_df) == 0:
        return False, {'reason': 'empty', 'data_coverage': 0, 'max_gap': float('inf'), 'actual_rate': 0}
    
    expected_samples = window_seconds * sampling_rate
    actual_samples = len(window_data_df)
    data_coverage = actual_samples / expected_samples
    
    if data_coverage < min_data_threshold:
        return False, {
            'reason': 'insufficient_data', 
            'data_coverage': data_coverage,
            'max_gap': float('inf'),
            'actual_rate': 0
        }
    
    if len(window_data_df) > 1:
        timestamps = pd.to_datetime(window_data_df['Timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds().fillna(0)
        max_gap = time_diffs.max()
        actual_rate = len(window_data_df) / (timestamps.max() - timestamps.min()).total_seconds()
    else:
        max_gap = 0
        actual_rate = sampling_rate
    
    if max_gap > max_gap_seconds:
        return False, {
            'reason': 'large_gap', 
            'data_coverage': data_coverage,
            'max_gap': max_gap,
            'actual_rate': actual_rate
        }
    
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
    """Versión robusta de remuestreo con múltiples estrategias"""
    if len(sensor_data) == 0:
        return np.zeros((target_timesteps, 3))
    
    original_timesteps = len(sensor_data)
    
    if original_timesteps == target_timesteps:
        return sensor_data.copy()
    
    if original_timesteps == 1:
        return np.tile(sensor_data[0], (target_timesteps, 1))
    
    try:
        if hasattr(timestamps[0], 'timestamp'):
            time_seconds = np.array([t.timestamp() for t in timestamps])
        elif isinstance(timestamps[0], pd.Timestamp):
            time_seconds = np.array([t.timestamp() for t in timestamps])
        else:
            time_seconds = timestamps.astype('int64') / 1e9
        
        time_min = time_seconds.min()
        time_max = time_seconds.max()
        
        if time_max > time_min:
            relative_times = (time_seconds - time_min) / (time_max - time_min)
        else:
            relative_times = np.linspace(0, 1, len(time_seconds))
        
        target_times = np.linspace(0, 1, target_timesteps)
        resampled_data = np.zeros((target_timesteps, 3))
        
        for axis in range(3):
            try:
                if original_timesteps >= target_timesteps:
                    resampled_axis = signal.resample(sensor_data[:, axis], target_timesteps)
                else:
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
                
            except Exception:
                resampled_data[:, axis] = np.interp(
                    target_times, relative_times, sensor_data[:, axis]
                )
        
        return resampled_data
    
    except Exception:
        return np.tile(sensor_data[0], (target_timesteps, 1))


def is_window_quality_good(resampled_window, max_std_threshold=50.0):
    """Verifica la calidad final de una ventana remuestreada"""
    if np.any(np.isnan(resampled_window)) or np.any(np.isinf(resampled_window)):
        return False
    
    if np.any(np.abs(resampled_window) > 1000):
        return False
    
    for axis in range(resampled_window.shape[1]):
        std_axis = np.std(resampled_window[:, axis])
        if std_axis > max_std_threshold or std_axis < 0.001:
            return False
    
    return True


def create_multimodal_windows_robust(df_accel, df_gyro=None, window_seconds=5, 
                                   overlap_percent=50, sampling_rate=20, 
                                   target_timesteps=250, min_data_threshold=0.8, 
                                   max_gap_seconds=1.0, sync_tolerance_ms=50):
    """
    Versión MULTIMODAL ROBUSTA: Crea ventanas sincronizadas de acelerómetro y giroscopio
    
    Args:
        df_accel: DataFrame con datos de acelerómetro (Polars o Pandas)
        df_gyro: DataFrame con datos de giroscopio (opcional)
        window_seconds: Duración de la ventana en segundos (default: 5)
        overlap_percent: Porcentaje de solapamiento (default: 50)
        sampling_rate: Frecuencia de muestreo en Hz (default: 20)
        target_timesteps: Número objetivo de timesteps por ventana (default: 250)
        min_data_threshold: Umbral mínimo de datos válidos (0.5 = 50%)
        max_gap_seconds: Máximo gap permitido en segundos (1.0s)
        sync_tolerance_ms: Tolerancia de sincronización en milisegundos (50ms)
        
    Returns:
        X: Array con forma (n_windows, timesteps, channels) - datos de ventanas
        y: Array con etiquetas de actividad
        subjects: Array con IDs de usuario
        metadata: DataFrame con información de las ventanas
    """
    
    # Determinar número de canales objetivo
    if df_gyro is not None:
        target_channels = 6  # 3 accel + 3 gyro
        print(f"🔧 CONFIGURACIÓN MULTIMODAL (Accel + Gyro):")
        mode = 'multimodal'
    else:
        target_channels = 3  # Solo accel
        print(f"🔧 CONFIGURACIÓN MONOMODAL (Solo Accel):")
        mode = 'monomodal'
    
    print(f"  Duración: {window_seconds}s")
    print(f"  Timesteps objetivo: {target_timesteps}")
    print(f"  Canales objetivo: {target_channels}")
    print(f"  Frecuencia de muestreo: {sampling_rate}Hz")
    print(f"  Solapamiento: {overlap_percent}%")
    print(f"  Umbral mínimo de datos: {min_data_threshold*100:.1f}%")
    print(f"  Máximo gap permitido: {max_gap_seconds}s")
    print(f"  Tolerancia sincronización: {sync_tolerance_ms}ms")
    
    # Preparar DataFrames
    df_accel_clean = prepare_sensor_dataframe(df_accel, 'accel')
    
    if df_gyro is not None:
        df_gyro_clean = prepare_sensor_dataframe(df_gyro, 'gyro')
        
        # Sincronizar datasets si ambos están disponibles
        print(f"\n🔄 SINCRONIZANDO SENSORES...")
        df_accel_sync, df_gyro_sync = synchronize_multimodal_data(
            df_accel_clean, df_gyro_clean, sync_tolerance_ms
        )
    else:
        df_accel_sync = df_accel_clean
        df_gyro_sync = None
    
    # Calcular parámetros temporales
    window_duration_ns = int(window_seconds * 1e9)
    step_duration_ns = int(window_duration_ns * (100 - overlap_percent) / 100)
    
    print(f"\n📏 PARÁMETROS TEMPORALES:")
    print(f"  Duración de ventana: {window_seconds}s")
    print(f"  Paso entre ventanas: {step_duration_ns / 1e9:.2f}s")
    
    # Almacenar resultados con forma consistente
    X_windows = []
    y_labels = []
    subjects_list = []
    metadata_list = []
    
    total_windows_attempted = 0
    total_windows_created = 0
    windows_with_gyro = 0
    windows_accel_only = 0
    
    # Procesar por usuario y actividad
    for (user_id, activity), accel_group in df_accel_sync.groupby(['Subject-id', 'Activity Label']):
        
        # Obtener grupo correspondiente de giroscopio
        if df_gyro_sync is not None:
            gyro_group = df_gyro_sync[
                (df_gyro_sync['Subject-id'] == user_id) & 
                (df_gyro_sync['Activity Label'] == activity)
            ]
            if len(gyro_group) == 0:
                print(f"⚠️ Usuario {user_id}, {activity}: Sin datos de giroscopio correspondientes")
                gyro_group = None
        else:
            gyro_group = None
        
        print(f"\n👤 Usuario {user_id}, {activity}:")
        print(f"   Accel: {len(accel_group)} muestras")
        if gyro_group is not None:
            print(f"   Gyro: {len(gyro_group)} muestras")
        
        # Verificar datos mínimos
        min_samples = window_seconds * sampling_rate
        if len(accel_group) < min_samples:
            print(f"   ⚠️ Muy pocos datos de acelerómetro ({len(accel_group)} < {min_samples})")
            continue
        
        # Crear ventanas multimodales
        windows_data = create_synchronized_windows_robust(
            accel_group, gyro_group, window_seconds, overlap_percent,
            target_timesteps, min_data_threshold, max_gap_seconds,
            sync_tolerance_ms, sampling_rate, target_channels, mode
        )
        
        # Procesar ventanas creadas
        window_count = 0
        for window_data in windows_data:
            total_windows_attempted += 1
            
            if window_data['is_valid']:
                # VERIFICAR FORMA CONSISTENTE
                window_shape = window_data['sensor_data'].shape
                expected_shape = (target_timesteps, target_channels)
                
                if window_shape == expected_shape:
                    X_windows.append(window_data['sensor_data'])
                    y_labels.append(activity)
                    subjects_list.append(user_id)
                    
                    # Contar tipos de ventana
                    if window_data['sensor_data'].shape[1] == 6:
                        windows_with_gyro += 1
                    else:
                        windows_accel_only += 1
                    
                    # Metadata detallada
                    metadata_list.append({
                        'Subject-id': user_id,
                        'Activity Label': activity,
                        'window_start': window_data['start_time'],
                        'window_end': window_data['end_time'],
                        'window_idx': window_count,
                        'accel_samples': window_data['accel_samples'],
                        'gyro_samples': window_data.get('gyro_samples', 0),
                        'sync_quality': window_data.get('sync_quality', 1.0),
                        'channels': target_channels,
                        'actual_channels': window_data['sensor_data'].shape[1],
                        'data_coverage': window_data['data_coverage'],
                        'max_gap_s': window_data['max_gap'],
                        'resampled_timesteps': target_timesteps
                    })
                    
                    window_count += 1
                    total_windows_created += 1
                else:
                    print(f"   ⚠️ Ventana con forma incorrecta: {window_shape} vs {expected_shape}")
        
        print(f"   ✅ Creadas {window_count} ventanas válidas")
    
    # Resumen y resultados
    print(f"\n📊 RESUMEN MULTIMODAL:")
    print(f"  Ventanas intentadas: {total_windows_attempted}")
    print(f"  Ventanas creadas: {total_windows_created}")
    print(f"  Ventanas con gyro: {windows_with_gyro}")
    print(f"  Ventanas solo accel: {windows_accel_only}")
    if total_windows_attempted > 0:
        print(f"  Tasa de éxito: {(total_windows_created/total_windows_attempted)*100:.1f}%")
    
    if len(X_windows) > 0:
        # VERIFICAR CONSISTENCIA ANTES DE CREAR ARRAY
        shapes = [w.shape for w in X_windows]
        unique_shapes = list(set(shapes))
        
        if len(unique_shapes) > 1:
            print(f"⚠️ ADVERTENCIA: Formas inconsistentes detectadas: {unique_shapes}")
            print("Filtrando solo ventanas con forma correcta...")
            
            # Filtrar solo ventanas con la forma correcta
            correct_shape = (target_timesteps, target_channels)
            valid_indices = [i for i, shape in enumerate(shapes) if shape == correct_shape]
            
            X_windows = [X_windows[i] for i in valid_indices]
            y_labels = [y_labels[i] for i in valid_indices]
            subjects_list = [subjects_list[i] for i in valid_indices]
            metadata_list = [metadata_list[i] for i in valid_indices]
            
            print(f"Ventanas filtradas: {len(X_windows)}")
        
        if len(X_windows) > 0:
            X = np.array(X_windows)
            y = np.array(y_labels)
            subjects = np.array(subjects_list)
            metadata_df = pd.DataFrame(metadata_list)
            
            print(f"\n✅ RESULTADO FINAL MULTIMODAL:")
            print(f"  Forma de X: {X.shape} (samples, timesteps, channels)")
            print(f"  Canales: {X.shape[2]} ({'accel_xyz + gyro_xyz' if target_channels == 6 else 'solo accel_xyz'})")
            print(f"  Total ventanas: {len(X)}")
            print(f"  Usuarios únicos: {len(np.unique(subjects))}")
            print(f"  Actividades: {sorted(np.unique(y))}")
            
            return X, y, subjects, metadata_df
        else:
            print("❌ No quedaron ventanas válidas después del filtrado")
            return None, None, None, None
    else:
        print("❌ No se crearon ventanas válidas")
        return None, None, None, None


def create_synchronized_windows_robust(accel_group, gyro_group, window_seconds, 
                                     overlap_percent, target_timesteps, 
                                     min_data_threshold, max_gap_seconds,
                                     sync_tolerance_ms, sampling_rate, 
                                     target_channels, mode):
    """Crea ventanas sincronizadas de múltiples sensores con forma consistente"""
    
    windows_data = []
    
    # Parámetros temporales
    window_duration_ns = int(window_seconds * 1e9)
    step_duration_ns = int(window_duration_ns * (100 - overlap_percent) / 100)
    
    # Rango temporal
    accel_times_ns = accel_group['Timestamp'].astype('int64')
    start_time_ns = accel_times_ns.min()
    end_time_ns = accel_times_ns.max()
    
    if gyro_group is not None:
        gyro_times_ns = gyro_group['Timestamp'].astype('int64')
        start_time_ns = max(start_time_ns, gyro_times_ns.min())
        end_time_ns = min(end_time_ns, gyro_times_ns.max())
    
    # Crear ventanas deslizantes
    current_start_ns = start_time_ns
    
    while current_start_ns + window_duration_ns <= end_time_ns:
        current_end_ns = current_start_ns + window_duration_ns
        
        # Extraer datos de acelerómetro para esta ventana
        accel_mask = (
            (accel_times_ns >= current_start_ns) & 
            (accel_times_ns < current_end_ns)
        )
        window_accel = accel_group[accel_mask]
        
        # Extraer datos de giroscopio si está disponible
        if gyro_group is not None:
            gyro_mask = (
                (gyro_times_ns >= current_start_ns) & 
                (gyro_times_ns < current_end_ns)
            )
            window_gyro = gyro_group[gyro_mask]
        else:
            window_gyro = None
        
        # Validar y procesar ventana CON FORMA CONSISTENTE
        window_data = process_multimodal_window_consistent(
            window_accel, window_gyro, target_timesteps, 
            window_seconds, min_data_threshold, max_gap_seconds,
            current_start_ns, current_end_ns, sampling_rate,
            target_channels, mode
        )
        
        windows_data.append(window_data)
        current_start_ns += step_duration_ns
    
    return windows_data


def process_multimodal_window_consistent(window_accel, window_gyro, target_timesteps,
                                       window_seconds, min_data_threshold, max_gap_seconds,
                                       start_time_ns, end_time_ns, sampling_rate,
                                       target_channels, mode):
    """
    Procesa una ventana multimodal con forma CONSISTENTE
    Siempre devuelve target_channels canales, rellenando con ceros si es necesario
    """
    
    # Validar acelerómetro
    accel_valid, accel_info = validate_window_data(
        window_accel, window_seconds, sampling_rate, min_data_threshold, max_gap_seconds
    )
    
    if not accel_valid:
        return {
            'is_valid': False,
            'reason': f"accel_{accel_info['reason']}",
            'start_time': pd.to_datetime(start_time_ns),
            'end_time': pd.to_datetime(end_time_ns),
            'accel_samples': len(window_accel),
            'gyro_samples': len(window_gyro) if window_gyro is not None else 0,
            'sync_quality': 0.0,
            'data_coverage': accel_info['data_coverage'],
            'max_gap': accel_info['max_gap']
        }
    
    # Procesar datos de acelerómetro
    accel_data = window_accel[['X', 'Y', 'Z']].values
    accel_timestamps = window_accel['Timestamp'].values
    
    try:
        accel_resampled = resample_window_robust(
            accel_data, accel_timestamps, target_timesteps, window_seconds
        )
    except Exception as e:
        return {
            'is_valid': False,
            'reason': f"accel_resample_error: {str(e)[:50]}",
            'start_time': pd.to_datetime(start_time_ns),
            'end_time': pd.to_datetime(end_time_ns),
            'accel_samples': len(window_accel),
            'gyro_samples': len(window_gyro) if window_gyro is not None else 0,
            'sync_quality': 0.0,
            'data_coverage': accel_info['data_coverage'],
            'max_gap': accel_info['max_gap']
        }
    
    # CREAR ARRAY CON FORMA CONSISTENTE
    final_data = np.zeros((target_timesteps, target_channels))
    
    # Copiar datos de acelerómetro (primeras 3 columnas)
    final_data[:, 0:3] = accel_resampled
    
    # Procesar giroscopio si está disponible y se requiere
    gyro_success = False
    gyro_samples = 0
    
    if target_channels == 6 and window_gyro is not None and len(window_gyro) > 0:
        gyro_valid, gyro_info = validate_window_data(
            window_gyro, window_seconds, sampling_rate, min_data_threshold, max_gap_seconds
        )
        
        if gyro_valid:
            gyro_data = window_gyro[['X', 'Y', 'Z']].values
            gyro_timestamps = window_gyro['Timestamp'].values
            gyro_samples = len(window_gyro)
            
            try:
                gyro_resampled = resample_window_robust(
                    gyro_data, gyro_timestamps, target_timesteps, window_seconds
                )
                
                # Verificar calidad del giroscopio
                if is_window_quality_good(gyro_resampled):
                    # Copiar datos de giroscopio (columnas 3-5)
                    final_data[:, 3:6] = gyro_resampled
                    gyro_success = True
                
            except Exception as e:
                # Si falla el giroscopio, las columnas 3-5 quedan en ceros
                pass
    
    # Verificar calidad final del acelerómetro
    if is_window_quality_good(accel_resampled):
        # Calcular sync_quality basado en disponibilidad de datos
        if target_channels == 6:
            sync_quality = 1.0 if gyro_success else 0.5  # Multimodal con/sin gyro
        else:
            sync_quality = 1.0  # Monomodal siempre 1.0
        
        return {
            'is_valid': True,
            'sensor_data': final_data,
            'start_time': pd.to_datetime(start_time_ns),
            'end_time': pd.to_datetime(end_time_ns),
            'accel_samples': len(window_accel),
            'gyro_samples': gyro_samples,
            'sync_quality': sync_quality,
            'data_coverage': accel_info['data_coverage'],
            'max_gap': accel_info['max_gap']
        }
    else:
        return {
            'is_valid': False,
            'reason': 'poor_accel_quality_after_resampling',
            'start_time': pd.to_datetime(start_time_ns),
            'end_time': pd.to_datetime(end_time_ns),
            'accel_samples': len(window_accel),
            'gyro_samples': gyro_samples,
            'sync_quality': 0.0,
            'data_coverage': accel_info['data_coverage'],
            'max_gap': accel_info['max_gap']
        }