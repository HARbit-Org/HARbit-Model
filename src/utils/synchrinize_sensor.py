import numpy as np
import polars as pl

def synchronize_sensors(df_gyro, df_accel, tolerance_ms=25):
    """
    Sincroniza sensores usando ventana de tolerancia temporal
    
    Args:
        tolerance_ms: Tolerancia en milisegundos (para 20Hz ≈ 50ms, usamos 25ms)
    """
    
    # Función para encontrar timestamps más cercanos
    def find_closest_timestamps(gyro_group, accel_group, tolerance_ns):
        gyro_times = gyro_group['Timestamp'].to_numpy()
        accel_times = accel_group['Timestamp'].to_numpy()
        
        matched_pairs = []
        used_accel_idx = set()
        
        for i, gyro_time in enumerate(gyro_times):
            # Buscar el timestamp de accel más cercano
            time_diffs = np.abs(accel_times - gyro_time)
            closest_idx = np.argmin(time_diffs)
            
            # Verificar si está dentro de la tolerancia y no usado
            if (time_diffs[closest_idx] <= tolerance_ns and 
                closest_idx not in used_accel_idx):
                
                matched_pairs.append({
                    'gyro_idx': i,
                    'accel_idx': closest_idx,
                    'timestamp': gyro_time,  # Usar timestamp del giroscopio como referencia
                    'time_diff_ms': time_diffs[closest_idx] / 1e6  # Convertir a ms
                })
                used_accel_idx.add(closest_idx)
        
        return matched_pairs
    
    tolerance_ns = tolerance_ms * 1e6  # Convertir a nanosegundos
    
    # Agrupar por sujeto y actividad
    gyro_grouped = df_gyro.group_by(['Subject-id', 'Activity Label'])
    accel_grouped = df_accel.group_by(['Subject-id', 'Activity Label'])
    
    synchronized_data = []
    
    for (subject_id, activity), gyro_group in gyro_grouped:
        # Buscar grupo correspondiente en accel
        accel_group = accel_grouped.get_group((subject_id, activity))
        
        if accel_group is not None:
            matches = find_closest_timestamps(
                gyro_group, accel_group, tolerance_ns
            )
            
            for match in matches:
                gyro_row = gyro_group[match['gyro_idx']]
                accel_row = accel_group[match['accel_idx']]
                
                synchronized_data.append({
                    'Subject-id': subject_id,
                    'Activity Label': activity,
                    'Timestamp': match['timestamp'],
                    'time_diff_ms': match['time_diff_ms'],
                    
                    # Datos del giroscopio
                    'gyro_X': gyro_row['X'],
                    'gyro_Y': gyro_row['Y'], 
                    'gyro_Z': gyro_row['Z'],
                    
                    # Datos del acelerómetro
                    'accel_X': accel_row['X'],
                    'accel_Y': accel_row['Y'],
                    'accel_Z': accel_row['Z']
                })
    
    return pl.DataFrame(synchronized_data)