from loguru import logger
from io import StringIO
import polars as pl
import pandas as pd
import os

def normalize_columns(data_base: pl.DataFrame | pd.DataFrame, 
                      user_col_name : str, timestamp_col_name: str, label_col_name: str, 
                      x_col_name: str, y_col_name: str, z_col_name: str):
    if isinstance(data_base, pl.DataFrame):
        df_normalize = data_base.select(
            pl.col(user_col_name).alias('Subject-id'),
            pl.col(timestamp_col_name).alias('Timestamp'),
            pl.col(label_col_name).alias('Activity Label'),
            pl.col(x_col_name).alias('X'),
            pl.col(y_col_name).alias('Y'),
            pl.col(z_col_name).alias('Z')
        )

        return df_normalize
    
    if isinstance(data_base, pd.DataFrame):
        df_normalize = data_base.rename( columns = {
            user_col_name: 'Subject-id',
            timestamp_col_name: 'Timestamp',
            label_col_name: 'Activity Label',
            x_col_name: 'X',
            y_col_name: 'Y',
            z_col_name: 'Z'
        })

        return df_normalize

    return None

def convert_timestamp(data_temp: pd.DataFrame | pl.DataFrame, type = 'ns'):
    if isinstance(data_temp, pl.DataFrame):
        # Caso Polars
        df_final = data_temp.with_columns(
                pl.col("Timestamp").cast(pl.Datetime(type)).alias("Timestamp")
            ).sort(["Subject-id", "Activity Label", "Timestamp"])
        
        return df_final
    
    if isinstance(data_temp, pd.DataFrame):
        # Caso Pandas
        df_final = data_temp.copy()
        df_final["Timestamp"] = pd.to_datetime(df_final["Timestamp"], errors="coerce")
        df_final = df_final.sort_values(["Subject-id", "Activity Label", "Timestamp"])

        return df_final
    
    return None

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
