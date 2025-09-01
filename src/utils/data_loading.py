from loguru import logger
from io import StringIO
import polars as pl
import os

# Carga de datos almacenados en archivos txt
def load_sensors_separately(path_base):
    """
    Carga datos de cada sensor por separado manteniendo identificación
    """
    sensors = ['gyro', 'accel']
    data_dict = {}
    
    for sensor in sensors:
        logger.info(f"Cargando datos de {sensor}...")
        sensor_data = ""
        
        path_dir = os.path.join(path_base, sensor)
        
        for txt_file in os.listdir(path_dir):
            if txt_file.endswith('.txt'):
                path_data = os.path.join(path_dir, txt_file)
                
                with open(path_data, 'r') as file:
                    content = file.read().replace(';', '')
                    sensor_data += (content + "\n")

        # Leer archivo individual
        df_temp = pl.read_csv(
            StringIO(sensor_data),
            schema={
                "Subject-id": pl.Float64, 
                "Activity Label": pl.Utf8,
                "Timestamp": pl.Int64,
                "X": pl.Float64,
                "Y": pl.Float64,
                "Z": pl.Float64
            },
            separator=','
        )

        # Convertir timestamp y ordenar
        df_final = df_temp.with_columns(
            pl.col('Timestamp').cast(pl.Datetime('ns')).alias('Timestamp')
        ).sort(['Subject-id', 'Activity Label', 'Timestamp'])
        
        df_final = df_final.filter(~pl.all_horizontal(pl.all().is_null()))

        data_dict[sensor] = df_final
    
    return data_dict
