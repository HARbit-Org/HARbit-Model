import pandas as pd
import polars as pl

def convert_frequences(data: pl.DataFrame, hz : int = 20):

    # Validar entrada
    if data.is_empty():
        return data
    
    # Ordenar datos
    data = data.sort(by=['Subject-id', 'Activity Label', 'Timestamp'])
    
    T = int((1 / hz) * 1000)
    str_frequence = str(T) + "ms"
    
    print(f"🔄 Convirtiendo a {hz}Hz (ventanas de {str_frequence})")
    
    # Lista para almacenar resultados
    results = []
    
    # Iterar por cada usuario
    for it, subject in enumerate(data["Subject-id"].unique()):
        print(f"📊 Procesando usuario {it+1}/{data['Subject-id'].n_unique()}: {subject}")
        
        # Filtrar solo ese usuario
        df_user = data.filter(pl.col("Subject-id") == subject)
        
        # Agrupar también por actividad para evitar mezclas
        for activity in df_user["Activity Label"].unique():
            df_user_activity = df_user.filter(pl.col("Activity Label") == activity)
            
            if df_user_activity.height == 0:
                continue
                
            print(f"  🎯 Actividad: {activity} ({df_user_activity.height} muestras)")
            
            # Hacer el resample a la frecuencia deseada
            try: 
                df_resampled = (
                    df_user_activity
                    .sort("Timestamp")
                    .group_by_dynamic("Timestamp", every=str_frequence)
                    .agg([
                        pl.mean("X").alias("X"),
                        pl.mean("Y").alias("Y"),
                        pl.mean("Z").alias("Z"),
                        pl.first("Subject-id").alias("Subject-id"),
                        pl.first("Activity Label").alias("Activity Label"),  # Más simple y seguro
                        pl.count().alias("samples_in_window")  # Para debug
                    ])
                    .filter(pl.col("samples_in_window") > 0)  # Filtrar ventanas vacías
                    .drop("samples_in_window")  # Remover columna auxiliar
                )
                
                print(f"    ✅ {df_user_activity.height} → {df_resampled.height} muestras")
                results.append(df_resampled)
                
            except Exception as e:
                print(f"    ❌ Error procesando {subject}-{activity}: {e}")
                continue
    
    # Concatenar todos los resultados
    if results:
        df_result = pl.concat(results)
        print(f"✅ Conversión completada: {data.height} → {df_result.height} muestras")
        return df_result.sort(by=['Subject-id', 'Activity Label', 'Timestamp'])
    else:
        print("❌ No se procesó ningún dato")
        return pl.DataFrame(schema=data.schema)