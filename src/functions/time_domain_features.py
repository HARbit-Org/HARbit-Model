import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

def extract_time_domain_features(window_data):
    """
    Extrae características del dominio del tiempo
    """
    features = {}
    
    for axis in ['X', 'Y', 'Z']:
        data = window_data[axis].to_numpy()
        
        # Características básicas
        features[f'{axis}_mean'] = np.mean(data)
        features[f'{axis}_std'] = np.std(data)
        features[f'{axis}_var'] = np.var(data)
        features[f'{axis}_min'] = np.min(data)
        features[f'{axis}_max'] = np.max(data)
        features[f'{axis}_range'] = np.max(data) - np.min(data)
        
        # Características estadísticas avanzadas
        features[f'{axis}_skewness'] = stats.skew(data)
        features[f'{axis}_kurtosis'] = stats.kurtosis(data)
        features[f'{axis}_rms'] = np.sqrt(np.mean(data**2))  # Root Mean Square
        features[f'{axis}_energy'] = np.sum(data**2)  # Energía
        
        # Características de la señal
        features[f'{axis}_mad'] = np.mean(np.abs(data - np.mean(data)))  # Mean Absolute Deviation
        features[f'{axis}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)  # Interquartile Range

    
        
    # Características inter-eje
    magnitude = np.sqrt(window_data['X']**2 + window_data['Y']**2 + window_data['Z']**2)
    features['magnitude_mean'] = np.mean(magnitude)
    features['magnitude_std'] = np.std(magnitude)
    
    return features