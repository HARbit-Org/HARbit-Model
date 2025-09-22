import pandas as pd
import polars as pl
import numpy as np

# Revisar si se puede optimizar con polars
def remove_start_end_seconds_advanced(data: pl.DataFrame, start_seconds: float = 0, 
                                    end_seconds: float = 0, min_segment_duration: float = None,
                                    timestamp_col: str = 'Timestamp', user_col: str = 'Subject-id', 
                                    label_col: str = 'Activity Label') -> pl.DataFrame:
    """
    Versión avanzada que considera gaps temporales y segmentos continuos
    
    Args:
        data: DataFrame con datos de sensores
        start_seconds: Segundos a eliminar del inicio de cada segmento continuo
        end_seconds: Segundos a eliminar del final de cada segmento continuo
        min_segment_duration: Duración mínima requerida para procesar un segmento
        timestamp_col: Nombre de la columna de timestamp
        user_col: Nombre de la columna de usuario
        label_col: Nombre de la columna de actividad
    
    Returns:
        DataFrame filtrado
    """
    
    if data.is_empty():
        print("⚠️ DataFrame vacío, retornando sin cambios")
        return data
    
    if min_segment_duration is None:
        min_segment_duration = start_seconds + end_seconds + 5  # Mínimo 5s adicionales
    
    print(f"🔄 ELIMINACIÓN AVANZADA: {start_seconds}s inicio, {end_seconds}s final")
    print(f"📊 Duración mínima de segmento: {min_segment_duration}s")
    
    # Verificar que timestamp es datetime
    if data[timestamp_col].dtype != pl.Datetime:
        data = data.with_columns(pl.col(timestamp_col).str.to_datetime())
    
    results = []
    segment_stats = []
    
    # Agrupar por usuario y actividad
    grouped = data.group_by([user_col, label_col])
    
    for group_key, group_data in grouped:
        user, activity = group_key
        
        # Ordenar por timestamp
        sorted_group = group_data.sort(timestamp_col)
        
        # Detectar segmentos continuos (gaps > 5 segundos indican nuevo segmento)
        timestamps = sorted_group[timestamp_col].to_list()
        
        if len(timestamps) < 2:
            continue
        
        # Calcular diferencias temporales
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                     for i in range(len(timestamps)-1)]
        
        # Encontrar puntos de ruptura (gaps > 5 segundos)
        gap_threshold = 5.0
        break_points = [0]  # Siempre empezar desde 0
        
        for i, diff in enumerate(time_diffs):
            if diff > gap_threshold:
                break_points.append(i + 1)
        
        break_points.append(len(timestamps))  # Punto final
        
        # Procesar cada segmento continuo
        for seg_idx in range(len(break_points) - 1):
            start_idx = break_points[seg_idx]
            end_idx = break_points[seg_idx + 1]
            
            segment = sorted_group[start_idx:end_idx]
            
            if segment.height < 10:
                continue
            
            # Calcular duración del segmento
            seg_start_time = segment[timestamp_col][0]
            seg_end_time = segment[timestamp_col][-1]
            segment_duration = (seg_end_time - seg_start_time).total_seconds()
            
            # Verificar duración mínima
            if segment_duration < min_segment_duration:
                print(f"  ⚠️ {user}-{activity} seg{seg_idx}: {segment_duration:.1f}s < {min_segment_duration}s (omitido)")
                continue
            
            # Calcular puntos de corte
            start_cutoff = seg_start_time + pl.duration(seconds=start_seconds)
            end_cutoff = seg_end_time - pl.duration(seconds=end_seconds)
            
            # Filtrar segmento
            filtered_segment = segment.filter(
                (pl.col(timestamp_col) >= start_cutoff) & 
                (pl.col(timestamp_col) <= end_cutoff)
            )
            
            if filtered_segment.height > 0:
                results.append(filtered_segment)
                
                segment_stats.append({
                    'user': user,
                    'activity': activity,
                    'segment': seg_idx,
                    'original_samples': segment.height,
                    'final_samples': filtered_segment.height,
                    'duration': segment_duration,
                    'samples_removed': segment.height - filtered_segment.height
                })
                
                print(f"  ✅ {user}-{activity} seg{seg_idx}: {segment.height} → {filtered_segment.height} "
                      f"({segment_duration:.1f}s)")
    
    # Mostrar estadísticas
    if segment_stats:
        total_original = sum(s['original_samples'] for s in segment_stats)
        total_final = sum(s['final_samples'] for s in segment_stats)
        total_removed = sum(s['samples_removed'] for s in segment_stats)
        
        print(f"\n📊 ESTADÍSTICAS DE SEGMENTACIÓN:")
        print(f"Segmentos continuos procesados: {len(segment_stats)}")
        print(f"Muestras: {total_original:,} → {total_final:,} (eliminadas: {total_removed:,})")
        print(f"Reducción: {(total_removed/total_original)*100:.2f}%")
    
    # Concatenar y retornar
    if results:
        return pl.concat(results).sort(by=[user_col, label_col, timestamp_col])
    else:
        print("❌ No se generaron segmentos válidos")
        return pl.DataFrame(schema=data.schema)