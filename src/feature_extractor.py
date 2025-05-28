import numpy as np
from scipy.fft import fft
import pandas as pd

class FeatureExtractor:
    def __init__(self, window_size=100, stride=50):
        self.window_size = window_size
        self.stride = stride

    def segment_data(self, data):
        """Create windows of data with specified size and stride."""
        segments = []
        labels = []
        subject_ids = []
        
        for subject_id in data['subject_id'].unique():
            subject_data = data[data['subject_id'] == subject_id]
            
            for i in range(0, len(subject_data) - self.window_size, self.stride):
                window = subject_data.iloc[i:i + self.window_size]
                
                # Get majority activity label for the window
                activity = window['activity'].mode()[0]
                
                segments.append(window)
                labels.append(activity)
                subject_ids.append(subject_id)
        
        return segments, labels, subject_ids

    def extract_statistical_features(self, window):
        """Extract statistical features from a window of data."""
        features = {}
        
        # Get sensor columns (accelerometer and gyroscope)
        sensor_cols = [col for col in window.columns if col.startswith(('x', 'y', 'z'))]
        
        for col in sensor_cols:
            data = window[col].values
            
            # Calculate statistical features
            features[f'{col}_mean'] = np.mean(data)
            features[f'{col}_std'] = np.std(data)
            features[f'{col}_min'] = np.min(data)
            features[f'{col}_max'] = np.max(data)
            features[f'{col}_rms'] = np.sqrt(np.mean(np.square(data)))
            features[f'{col}_energy'] = np.sum(np.square(data))
            
        # Calculate Signal Magnitude Area (SMA)
        for sensor in ['acc', 'gyro']:
            x = window[f'x_{sensor}'].values
            y = window[f'y_{sensor}'].values
            z = window[f'z_{sensor}'].values
            features[f'sma_{sensor}'] = (
                np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))
            ) / len(x)
        
        return features

    def extract_spectral_features(self, window):
        """Extract spectral features from a window of data."""
        features = {}
        
        sensor_cols = [col for col in window.columns if col.startswith(('x', 'y', 'z'))]
        
        for col in sensor_cols:
            data = window[col].values
            
            # Compute FFT
            fft_values = fft(data)
            fft_magnitude = np.abs(fft_values[:len(data)//2])
            
            # Extract features
            features[f'{col}_fft_peak'] = np.max(fft_magnitude)
            features[f'{col}_fft_freq_idx'] = np.argmax(fft_magnitude)
            
        return features

    def extract_features(self, data):
        """Extract all features from the data."""
        segments, labels, subject_ids = self.segment_data(data)
        
        # Extract features for each segment
        feature_vectors = []
        for segment in segments:
            # Get statistical features
            features = self.extract_statistical_features(segment)
            
            # Get spectral features
            spectral_features = self.extract_spectral_features(segment)
            features.update(spectral_features)
            
            feature_vectors.append(features)
        
        # Convert to DataFrame
        X = pd.DataFrame(feature_vectors)
        y = np.array(labels)
        groups = np.array(subject_ids)
        
        return X, y, groups 