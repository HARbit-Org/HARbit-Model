import pandas as pd
import polars as pl
import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

def combine_raw_and_features_batched(X_raw, X_features, mode='weighted_concat', 
                                   batch_size=5000, target_timesteps=100):
    """
    Combina datos en lotes para evitar problemas de memoria
    """
    n_samples = X_raw.shape[0]
    n_timesteps = X_raw.shape[1] 
    n_channels = X_raw.shape[2]
    n_features = X_features.shape[1]
    
    print(f"🔄 Procesando {n_samples} muestras en lotes de {batch_size}")
    print(f"📊 Memoria estimada: {(n_samples * n_timesteps * (n_channels + n_features) * 4) / 1e9:.2f} GB")
    
    # Crear array de salida
    final_shape = (n_samples, n_timesteps, n_channels + n_features)
    X_combined = np.empty(final_shape, dtype=np.float32)
    
    # Procesar por lotes
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        
        print(f"    Procesando lote {start_idx//batch_size + 1}: {start_idx}-{end_idx}")
        
        # Extraer lote
        X_raw_batch = X_raw[start_idx:end_idx]
        X_features_batch = X_features[start_idx:end_idx]
        
        # Combinar lote
        batch_combined = _combine_batch(X_raw_batch, X_features_batch, mode, n_timesteps)
        
        # Guardar en array final
        X_combined[start_idx:end_idx] = batch_combined
        
        # Limpiar memoria del lote
        del X_raw_batch, X_features_batch, batch_combined
    
    return X_combined

def _combine_batch(X_raw_batch, X_features_batch, mode, timesteps):
    """Función auxiliar para combinar un lote"""
    if mode == 'weighted_concat':
        # Normalizar
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_features_normalized = scaler.fit_transform(X_features_batch)
        
        # Escalar
        raw_std = np.std(X_raw_batch)
        feature_std = np.std(X_features_normalized)
        
        if feature_std > 0:
            X_features_scaled = X_features_normalized * (raw_std / feature_std)
        else:
            X_features_scaled = X_features_normalized
        
        # Expandir
        X_features_expanded = np.repeat(
            X_features_scaled[:, np.newaxis, :], 
            timesteps, 
            axis=1
        )
        
        # Concatenar
        return np.concatenate([X_raw_batch, X_features_expanded], axis=2)
    
    else:
        # Modo simple
        X_features_expanded = np.repeat(
            X_features_batch[:, np.newaxis, :], 
            timesteps, 
            axis=1
        )
        return np.concatenate([X_raw_batch, X_features_expanded], axis=2)


# ---------------------------------------------------------------------------------
#                               CREACION DE VENTANAS
# ---------------------------------------------------------------------------------

def extract_temporal_features(signal_data):
    """
    Extrae características del dominio temporal de una señal
    
    Args:
        signal_data: Array de forma (timesteps, channels) o (timesteps,)
    
    Returns:
        dict: Diccionario con características temporales
    """
    features = {}
    
    # Asegurar que sea 2D
    if signal_data.ndim == 1:
        signal_data = signal_data.reshape(-1, 1)
    
    n_timesteps, n_channels = signal_data.shape
    
    for ch in range(n_channels):
        channel_data = signal_data[:, ch]
        prefix = f'ch{ch}_'
        
        # Características básicas
        features[f'{prefix}mean'] = np.mean(channel_data)
        features[f'{prefix}std'] = np.std(channel_data)
        features[f'{prefix}var'] = np.var(channel_data)
        features[f'{prefix}min'] = np.min(channel_data)
        features[f'{prefix}max'] = np.max(channel_data)
        features[f'{prefix}range'] = features[f'{prefix}max'] - features[f'{prefix}min']
        features[f'{prefix}median'] = np.median(channel_data)
        
        # Percentiles
        features[f'{prefix}q25'] = np.percentile(channel_data, 25)
        features[f'{prefix}q75'] = np.percentile(channel_data, 75)
        features[f'{prefix}iqr'] = features[f'{prefix}q75'] - features[f'{prefix}q25']
        
        # Características de forma
        features[f'{prefix}skewness'] = skew(channel_data)
        features[f'{prefix}kurtosis'] = kurtosis(channel_data)
        
        # RMS (Root Mean Square)
        features[f'{prefix}rms'] = np.sqrt(np.mean(channel_data**2))
        
        # Energy
        features[f'{prefix}energy'] = np.sum(channel_data**2)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(channel_data)))
        features[f'{prefix}zcr'] = zero_crossings / len(channel_data)
        
        # Mean absolute deviation
        features[f'{prefix}mad'] = np.mean(np.abs(channel_data - np.mean(channel_data)))
        
        # Características de la primera diferencia (velocidad)
        if len(channel_data) > 1:
            diff_data = np.diff(channel_data)
            features[f'{prefix}diff_mean'] = np.mean(diff_data)
            features[f'{prefix}diff_std'] = np.std(diff_data)
            features[f'{prefix}diff_max'] = np.max(np.abs(diff_data))
            
            # Segunda diferencia (aceleración)
            if len(diff_data) > 1:
                diff2_data = np.diff(diff_data)
                features[f'{prefix}diff2_mean'] = np.mean(diff2_data)
                features[f'{prefix}diff2_std'] = np.std(diff2_data)
                features[f'{prefix}diff2_max'] = np.max(np.abs(diff2_data))
    
    # Características inter-canal si hay múltiples canales
    if n_channels > 1:
        # Correlación entre canales
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                corr = np.corrcoef(signal_data[:, i], signal_data[:, j])[0, 1]
                features[f'corr_ch{i}_ch{j}'] = corr if not np.isnan(corr) else 0.0
        
        # Magnitud vectorial (para datos de acelerómetro/giroscopio)
        if n_channels == 3:
            magnitude = np.sqrt(np.sum(signal_data**2, axis=1))
            features['magnitude_mean'] = np.mean(magnitude)
            features['magnitude_std'] = np.std(magnitude)
            features['magnitude_max'] = np.max(magnitude)
            features['magnitude_min'] = np.min(magnitude)
    
    return features


def extract_frequency_features(signal_data, sampling_rate=20):
    """
    Extrae características del dominio frecuencial usando FFT
    
    Args:
        signal_data: Array de forma (timesteps, channels) o (timesteps,)
        sampling_rate: Frecuencia de muestreo en Hz
    
    Returns:
        dict: Diccionario con características frecuenciales
    """
    features = {}
    
    # Asegurar que sea 2D
    if signal_data.ndim == 1:
        signal_data = signal_data.reshape(-1, 1)
    
    n_timesteps, n_channels = signal_data.shape
    
    for ch in range(n_channels):
        channel_data = signal_data[:, ch]
        prefix = f'ch{ch}_'
        
        # Aplicar ventana para reducir el leakage espectral
        windowed_data = channel_data * np.hanning(len(channel_data))
        
        # FFT
        fft_values = fft(windowed_data)
        fft_magnitude = np.abs(fft_values[:len(fft_values)//2])  # Solo frecuencias positivas
        fft_freqs = fftfreq(len(windowed_data), 1/sampling_rate)[:len(fft_values)//2]
        
        # Normalizar el espectro
        fft_magnitude = fft_magnitude / len(windowed_data)
        
        # Densidad espectral de potencia
        psd = fft_magnitude**2
        
        # Características básicas del espectro
        features[f'{prefix}spectral_energy'] = np.sum(psd)
        features[f'{prefix}spectral_mean'] = np.mean(fft_magnitude)
        features[f'{prefix}spectral_std'] = np.std(fft_magnitude)
        features[f'{prefix}spectral_max'] = np.max(fft_magnitude)
        
        # Frecuencia dominante
        dominant_freq_idx = np.argmax(psd[1:]) + 1  # Excluir DC
        features[f'{prefix}dominant_freq'] = fft_freqs[dominant_freq_idx]
        features[f'{prefix}dominant_freq_magnitude'] = fft_magnitude[dominant_freq_idx]
        
        # Centroide espectral (centro de masa del espectro)
        if np.sum(psd) > 0:
            spectral_centroid = np.sum(fft_freqs * psd) / np.sum(psd)
            features[f'{prefix}spectral_centroid'] = spectral_centroid
        else:
            features[f'{prefix}spectral_centroid'] = 0.0
        
        # Rolloff espectral (frecuencia por debajo de la cual está el 85% de la energía)
        cumulative_energy = np.cumsum(psd)
        total_energy = cumulative_energy[-1]
        if total_energy > 0:
            rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
            if len(rolloff_idx) > 0:
                features[f'{prefix}spectral_rolloff'] = fft_freqs[rolloff_idx[0]]
            else:
                features[f'{prefix}spectral_rolloff'] = fft_freqs[-1]
        else:
            features[f'{prefix}spectral_rolloff'] = 0.0
        
        # Spread espectral (dispersión alrededor del centroide)
        if np.sum(psd) > 0:
            spectral_spread = np.sqrt(np.sum(((fft_freqs - features[f'{prefix}spectral_centroid'])**2) * psd) / np.sum(psd))
            features[f'{prefix}spectral_spread'] = spectral_spread
        else:
            features[f'{prefix}spectral_spread'] = 0.0
        
        # Skewness y kurtosis espectrales
        features[f'{prefix}spectral_skewness'] = skew(fft_magnitude)
        features[f'{prefix}spectral_kurtosis'] = kurtosis(fft_magnitude)
        
        # Entropía espectral
        psd_normalized = psd / (np.sum(psd) + 1e-12)
        spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))
        features[f'{prefix}spectral_entropy'] = spectral_entropy
        
        # Características en bandas de frecuencia específicas
        # Banda baja (0-2 Hz)
        low_band_mask = (fft_freqs >= 0) & (fft_freqs <= 2)
        features[f'{prefix}low_band_energy'] = np.sum(psd[low_band_mask])
        
        # Banda media (2-5 Hz)
        mid_band_mask = (fft_freqs > 2) & (fft_freqs <= 5)
        features[f'{prefix}mid_band_energy'] = np.sum(psd[mid_band_mask])
        
        # Banda alta (5-10 Hz)
        high_band_mask = (fft_freqs > 5) & (fft_freqs <= 10)
        features[f'{prefix}high_band_energy'] = np.sum(psd[high_band_mask])
        
        # Ratios de energía entre bandas
        total_band_energy = features[f'{prefix}low_band_energy'] + features[f'{prefix}mid_band_energy'] + features[f'{prefix}high_band_energy']
        if total_band_energy > 0:
            features[f'{prefix}low_band_ratio'] = features[f'{prefix}low_band_energy'] / total_band_energy
            features[f'{prefix}mid_band_ratio'] = features[f'{prefix}mid_band_energy'] / total_band_energy
            features[f'{prefix}high_band_ratio'] = features[f'{prefix}high_band_energy'] / total_band_energy
        else:
            features[f'{prefix}low_band_ratio'] = 0.0
            features[f'{prefix}mid_band_ratio'] = 0.0
            features[f'{prefix}high_band_ratio'] = 0.0
    
    return features


def extract_combined_features(signal_data, sampling_rate=20):
    """
    Extrae características tanto temporales como frecuenciales
    
    Args:
        signal_data: Array de forma (timesteps, channels)
        sampling_rate: Frecuencia de muestreo en Hz
    
    Returns:
        dict: Diccionario combinado con todas las características
    """
    temporal_features = extract_temporal_features(signal_data)
    frequency_features = extract_frequency_features(signal_data, sampling_rate)
    
    # Combinar ambos diccionarios
    combined_features = {**temporal_features, **frequency_features}
    
    return combined_features


def create_multimodal_windows_with_features(df_accel, df_gyro=None, window_seconds=5, 
                                          overlap_percent=50, sampling_rate=20, 
                                          target_timesteps=250, min_data_threshold=0.8, 
                                          max_gap_seconds=1.0, sync_tolerance_ms=50,
                                          extract_features=True,
                                          fusion_strategy="weighted_concat"):
    """
    Versión EXTENDIDA: Crea ventanas con extracción opcional de características
    
    Args:
        extract_features: Si True, extrae características temporales y frecuenciales
        
    Returns:
        Si extract_features=True:
            X_raw: Array con datos crudos
            X_features: Array con características extraídas  
            y, subjects, metadata: Como antes
        Si extract_features=False:
            X, y, subjects, metadata: Solo datos crudos
    """
    
    # Llamar a la función original
    X_raw, y, subjects, metadata_df = create_multimodal_windows_robust(
        df_accel=df_accel,
        df_gyro=df_gyro,
        window_seconds=window_seconds,
        overlap_percent=overlap_percent,
        sampling_rate=sampling_rate,
        target_timesteps=target_timesteps,
        min_data_threshold=min_data_threshold,
        max_gap_seconds=max_gap_seconds,
        sync_tolerance_ms=sync_tolerance_ms
    )
    
    if X_raw is None or not extract_features:
        if extract_features:
            return X_raw, None, y, subjects, metadata_df
        else:
            return X_raw, y, subjects, metadata_df
    
    print(f"\n🔬 EXTRAYENDO CARACTERÍSTICAS AVANZADAS...")
    print(f"  Procesando {len(X_raw)} ventanas...")
    
    # Extraer características de cada ventana
    feature_list = []
    
    for i, window in enumerate(X_raw):
        if i % 1000 == 0:
            print(f"    Progreso: {i}/{len(X_raw)} ({100*i/len(X_raw):.1f}%)")
        
        try:
            # Extraer características combinadas
            window_features = extract_combined_features(window, sampling_rate)
            feature_list.append(window_features)
            
        except Exception as e:
            print(f"    ⚠️ Error en ventana {i}: {str(e)}")
            # Crear diccionario vacío con las mismas claves que una ventana válida
            if len(feature_list) > 0:
                empty_features = {key: 0.0 for key in feature_list[0].keys()}
                feature_list.append(empty_features)
            else:
                feature_list.append({})
    
    if len(feature_list) > 0 and len(feature_list[0]) > 0:
        # Convertir lista de diccionarios a array numpy
        feature_df = pd.DataFrame(feature_list)
        X_features = feature_df.values
        
        print(f"  ✅ Características extraídas:")
        print(f"    Forma de X_features: {X_features.shape}")
        print(f"    Características por ventana: {X_features.shape[1]}")
        
        # Agregar información de características a metadata
        if metadata_df is not None:
            metadata_df['n_features'] = X_features.shape[1]
            metadata_df['feature_extraction'] = True

        # X_combined = combine_raw_and_features_batched(
        #     X_raw,
        #     X_features,
        #     mode= fusion_strategy,
        #     target_timesteps=target_timesteps
        # )

        return X_raw, X_features, y, subjects, metadata_df
    else:
        print("  ❌ No se pudieron extraer características")
        return X_raw, None, y, subjects, metadata_df


# Actualizar la función principal para incluir características opcionales
def create_multimodal_windows_robust(df_accel, df_gyro=None, window_seconds=5, 
                                   overlap_percent=50, sampling_rate=20, 
                                   target_timesteps=250, min_data_threshold=0.8, 
                                   max_gap_seconds=1.0, sync_tolerance_ms=50):
    """
    Versión MULTIMODAL ROBUSTA: Crea ventanas sincronizadas de acelerómetro y giroscopio
    
    [Mantener la implementación original exactamente igual]
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