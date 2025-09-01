import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

def extract_frequency_domain_features(window_data, sampling_rate=20):
    """
    Extrae características del dominio de la frecuencia
    """
    features = {}
    
    for axis in ['X', 'Y', 'Z']:
        data = window_data[axis].to_numpy()
        
        # FFT
        fft_values = np.abs(fft(data))
        freqs = fftfreq(len(data), 1/sampling_rate)
        
        # Solo tomar la mitad positiva del espectro
        n = len(fft_values) // 2
        fft_values = fft_values[:n]
        freqs = freqs[:n]
        
        # Características espectrales
        features[f'{axis}_spectral_energy'] = np.sum(fft_values**2)
        features[f'{axis}_spectral_entropy'] = stats.entropy(fft_values + 1e-12)
        features[f'{axis}_dominant_freq'] = freqs[np.argmax(fft_values)]
        features[f'{axis}_spectral_centroid'] = np.sum(freqs * fft_values) / np.sum(fft_values)
        
        # Picos espectrales
        peaks, _ = find_peaks(fft_values, height=np.max(fft_values) * 0.1)
        features[f'{axis}_num_peaks'] = len(peaks)
        
        if len(peaks) > 0:
            features[f'{axis}_peak_freq_mean'] = np.mean(freqs[peaks])
            features[f'{axis}_peak_magnitude_mean'] = np.mean(fft_values[peaks])
        else:
            features[f'{axis}_peak_freq_mean'] = 0
            features[f'{axis}_peak_magnitude_mean'] = 0
            
        # Bandas de frecuencia
        low_band = (freqs >= 0) & (freqs < 2)
        mid_band = (freqs >= 2) & (freqs < 5)
        high_band = (freqs >= 5) & (freqs <= 10)
        
        features[f'{axis}_low_freq_energy'] = np.sum(fft_values[low_band]**2)
        features[f'{axis}_mid_freq_energy'] = np.sum(fft_values[mid_band]**2)
        features[f'{axis}_high_freq_energy'] = np.sum(fft_values[high_band]**2)
    
    return features