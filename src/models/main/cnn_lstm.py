import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler,
    label_binarize
)

import yaml

with open(r"F:\UPC\Tesis\HARbit-Model\src\models\config\hiperparameters.yaml", 'r') as file:
    config = yaml.safe_load(file)['config']

_cnn_lstm_config = config['cnn-lstm']

def create_cnn_lstm_model(input_shape, num_classes):
    """
    Modelo CNN-LSTM optimizado para secuencias de características extraídas.
    Diseñado para input_shape = (timesteps, features), ej: (10, 68).
    """
    model = Sequential([
        # Bloque convolucional: detecta patrones locales en las secuencias
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        # Dropout(0.3),

        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        # Dropout(0.3),

        MaxPooling1D(pool_size=2),  # reduce dimensionalidad y fuerza abstracción

        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),

        # LSTM: captura dependencias temporales en las features extraídas
        LSTM(128, return_sequences=False, dropout=0.4, recurrent_dropout=0.4),

        # Clasificación densa
        Dense(256, activation='relu'),
        BatchNormalization(),
        # Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        # Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    # Compilación
    model.compile(
        optimizer   = Adam(learning_rate=_cnn_lstm_config['learning_rate']),
        loss        = _cnn_lstm_config['loss'],
        metrics     = _cnn_lstm_config['metrics']
    )
    
    return model