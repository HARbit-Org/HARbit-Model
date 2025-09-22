from scipy.signal import butter, filtfilt, resample_poly
import pandas as pd
import polars as pl
import numpy as np

def convert_freq_antialias(data: pl.DataFrame, hz: int = 20, orig_hz: int = 100) -> pl.DataFrame:
    """
    Resampleo correcto con filtro anti-aliasing + decimación.
    Procesa por usuario y actividad de forma separada.
    """
    df = data.to_pandas()

    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    dec_factor = orig_hz // hz
    cutoff = hz / 2 * 0.8  # ligeramente menor que Nyquist
    b, a = butter(4, cutoff / (orig_hz / 2), btype="low")

    new_records = []
    for (subj, act), df_group in df.groupby(["Subject-id", "Activity Label"]):
        if df_group.shape[0] < dec_factor:
            continue

        X, Y, Z = df_group["X"].to_numpy(), df_group["Y"].to_numpy(), df_group["Z"].to_numpy()

        # filtro pasa-bajo
        X_f = filtfilt(b, a, X)
        Y_f = filtfilt(b, a, Y)
        Z_f = filtfilt(b, a, Z)

        # decimación
        X_ds = resample_poly(X_f, up=1, down=dec_factor)
        Y_ds = resample_poly(Y_f, up=1, down=dec_factor)
        Z_ds = resample_poly(Z_f, up=1, down=dec_factor)

        # timestamps nuevos (cada 1/hz segundos)
        start = df_group["Timestamp"].iloc[0]
        new_times = pd.date_range(start=start, periods=len(X_ds), freq=f"{int(1000/hz)}ms")

        new_records.append(pd.DataFrame({
            "Subject-id": subj,
            "Activity Label": act,
            "Timestamp": new_times,
            "X": X_ds,
            "Y": Y_ds,
            "Z": Z_ds
        }))

    if not new_records:
        return pl.DataFrame(schema=data.schema)

    df_final = pd.concat(new_records, ignore_index=True)
    return pl.from_pandas(df_final).sort(["Subject-id", "Activity Label", "Timestamp"])

def convert_frequences(data: pl.DataFrame, hz: int = 20, interpolation_method: str = "mean", 
                      min_samples_per_window: int = 1) -> pl.DataFrame:
    """
    Convierte la frecuencia de muestreo de datos de sensores usando resampling
    
    Args:
        data: DataFrame con columnas ['Subject-id', 'Activity Label', 'Timestamp', 'X', 'Y', 'Z']
        hz: Frecuencia objetivo en Hz
        interpolation_method: Método de agregación ('mean', 'median', 'first', 'last')
        min_samples_per_window: Mínimo de muestras por ventana para considerar válida
    
    Returns:
        DataFrame con frecuencia convertida
    """
    
    # Validación de entrada mejorada
    if data.is_empty():
        print("⚠️ DataFrame vacío, retornando sin cambios")
        return data
    
    required_columns = ['Subject-id', 'Activity Label', 'Timestamp', 'X', 'Y', 'Z']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"❌ Columnas faltantes: {missing_cols}")
    
    # Validación de parámetros
    if hz <= 0:
        raise ValueError("❌ La frecuencia debe ser mayor a 0")
    
    # Verificar que Timestamp es datetime
    if data['Timestamp'].dtype != pl.Datetime:
        print("🔄 Convirtiendo Timestamp a datetime...")
        data = data.with_columns(pl.col('Timestamp').str.to_datetime())
    
    # Ordenar datos
    data = data.sort(by=['Subject-id', 'Activity Label', 'Timestamp'])
    
    # Calcular intervalo de muestreo
    T = int((1 / hz) * 1000)
    str_frequence = f"{T}ms"
    
    print(f"🔄 Convirtiendo a {hz}Hz (ventanas de {str_frequence})")
    print(f"📊 Método de interpolación: {interpolation_method}")
    print(f"📊 Mínimo de muestras por ventana: {min_samples_per_window}")
    
    # Definir función de agregación según el método
    agg_funcs = {
        'mean': pl.mean,
        'median': pl.median,
        'first': pl.first,
        'last': pl.last
    }
    
    if interpolation_method not in agg_funcs:
        raise ValueError(f"❌ Método no válido. Use: {list(agg_funcs.keys())}")
    
    agg_func = agg_funcs[interpolation_method]
    
    # Lista para almacenar resultados
    results = []
    total_original = 0
    total_resampled = 0
    
    # Estadísticas de procesamiento
    processing_stats = {
        'users_processed': 0,
        'activities_processed': 0,
        'users_failed': 0,
        'activities_failed': 0
    }
    
    # Iterar por cada usuario
    unique_subjects = data["Subject-id"].unique().to_list()
    
    for it, subject in enumerate(unique_subjects):
        print(f"📊 Procesando usuario {it+1}/{len(unique_subjects)}: {subject}")
        
        # Filtrar solo ese usuario
        df_user = data.filter(pl.col("Subject-id") == subject)
        user_success = False
        
        # Agrupar también por actividad para evitar mezclas
        unique_activities = df_user["Activity Label"].unique().to_list()
        
        for activity in unique_activities:
            df_user_activity = df_user.filter(pl.col("Activity Label") == activity)
            
            if df_user_activity.height == 0:
                continue
                
            original_samples = df_user_activity.height
            total_original += original_samples
            
            print(f"  🎯 Actividad: {activity} ({original_samples} muestras)")
            
            # Verificar continuidad temporal
            timestamps = df_user_activity['Timestamp'].to_list()
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                         for i in range(len(timestamps)-1)]
            
            if time_diffs:
                avg_interval = np.mean(time_diffs)
                target_interval = 1/hz
                print(f"    📈 Intervalo promedio actual: {avg_interval:.3f}s (objetivo: {target_interval:.3f}s)")
            
            # Hacer el resample a la frecuencia deseada
            try: 
                df_resampled = (
                    df_user_activity
                    .sort("Timestamp")
                    .group_by_dynamic("Timestamp", every=str_frequence)
                    .agg([
                        agg_func("X").alias("X"),
                        agg_func("Y").alias("Y"),
                        agg_func("Z").alias("Z"),
                        pl.first("Subject-id").alias("Subject-id"),
                        pl.first("Activity Label").alias("Activity Label"),
                        pl.count().alias("samples_in_window"),
                        pl.first("Timestamp").alias("window_start"),
                        pl.last("Timestamp").alias("window_end")
                    ])
                    .filter(pl.col("samples_in_window") >= min_samples_per_window)
                    .with_columns([
                        # Usar el inicio de la ventana como timestamp representativo
                        pl.col("window_start").alias("Timestamp")
                    ])
                    .drop(["samples_in_window", "window_start", "window_end"])
                )
                
                resampled_samples = df_resampled.height
                total_resampled += resampled_samples
                
                compression_ratio = resampled_samples / original_samples if original_samples > 0 else 0
                print(f"    ✅ {original_samples} → {resampled_samples} muestras (ratio: {compression_ratio:.3f})")
                
                if resampled_samples > 0:
                    results.append(df_resampled)
                    processing_stats['activities_processed'] += 1
                    user_success = True
                else:
                    print(f"    ⚠️ No se generaron muestras válidas")
                    processing_stats['activities_failed'] += 1
                
            except Exception as e:
                print(f"    ❌ Error procesando {subject}-{activity}: {e}")
                processing_stats['activities_failed'] += 1
                continue
        
        if user_success:
            processing_stats['users_processed'] += 1
        else:
            processing_stats['users_failed'] += 1
    
    # Concatenar todos los resultados
    if results:
        df_result = pl.concat(results)
        
        # Estadísticas finales
        final_samples = df_result.height
        overall_compression = final_samples / total_original if total_original > 0 else 0
        
        print(f"\n✅ CONVERSIÓN COMPLETADA")
        print(f"📊 Muestras: {total_original:,} → {final_samples:,} (ratio: {overall_compression:.3f})")
        print(f"👥 Usuarios procesados: {processing_stats['users_processed']}/{len(unique_subjects)}")
        print(f"🎯 Actividades procesadas: {processing_stats['activities_processed']}")
        
        if processing_stats['users_failed'] > 0 or processing_stats['activities_failed'] > 0:
            print(f"⚠️ Fallos - Usuarios: {processing_stats['users_failed']}, Actividades: {processing_stats['activities_failed']}")
        
        return df_result.sort(by=['Subject-id', 'Activity Label', 'Timestamp'])
    else:
        print("❌ No se procesó ningún dato válido")
        return pl.DataFrame(schema=data.schema)