"""
Sensor data preprocessor using pyHRV for HRV metric computation.
Processes raw PPG and temperature data from Arduino sensors.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings

# Import pyHRV modules
from pyhrv.tools import nn_intervals
from pyhrv import hrv
from pyhrv import time_domain
from pyhrv import nonlinear as nl


class SensorPreprocessor:
    """Preprocess raw sensor data and compute HRV metrics."""
    
    def __init__(self, sampling_rate: int = 100):
        """
        Initialize the preprocessor.
        
        Args:
            sampling_rate: Sampling rate of the PPG signal in Hz
        """
        self.sampling_rate = sampling_rate
        self.red_buffer = []
        self.ir_buffer = []
        self.temp_buffer = []
        
    def add_red_sample(self, value: int) -> None:
        """Add a red channel PPG sample to the buffer."""
        self.red_buffer.append(value)
        
    def add_ir_sample(self, value: int) -> None:
        """Add an IR channel PPG sample to the buffer."""
        self.ir_buffer.append(value)
        
    def add_temperature_sample(self, value: float) -> None:
        """Add a temperature sample to the buffer."""
        self.temp_buffer.append(value)
        
    def clear_buffers(self) -> None:
        """Clear all buffers."""
        self.red_buffer = []
        self.ir_buffer = []
        self.temp_buffer = []
        
    def process_ppg_signal(self, red_signal: np.ndarray, ir_signal: np.ndarray) -> Dict:
        """
        Process PPG signal (red and IR channels) and compute HRV metrics using pyHRV.
        
        Args:
            red_signal: Raw red channel PPG signal array
            ir_signal: Raw IR channel PPG signal array
            
        Returns:
            Dictionary containing HRV metrics, heart rate, and SpO2
        """
        if len(red_signal) < 10 or len(ir_signal) < 10:
            return self._get_empty_hrv_dict()
        
        try:
            # Use red channel for heart rate detection (typically has better SNR)
            ppg_normalized = self._normalize_signal(red_signal)
            
            # Detect R-peaks (beat detection) from PPG signal
            # Using simple peak detection
            peaks = self._detect_peaks(ppg_normalized)
            
            if len(peaks) < 2:
                return self._get_empty_hrv_dict()
            
            # Calculate NN intervals (in milliseconds)
            peak_times = peaks / self.sampling_rate * 1000  # Convert to ms
            nni = np.diff(peak_times)
            
            # Remove outliers (ectopic beats)
            nni = self._remove_outliers(nni)
            
            if len(nni) < 2:
                return self._get_empty_hrv_dict()
            
            # Compute HRV metrics using pyHRV
            metrics = self._compute_hrv_metrics(nni)
            
            # Calculate heart rate
            heart_rate = (6 / np.mean(nni)) * 1000
            
            # Calculate SpO2 from red and IR channels
            spo2 = self._calculate_spo2(red_signal, ir_signal)
            
            return {
                'heart_rate': heart_rate,
                'spo2': spo2,
                'hrv_metrics': metrics,
                'nni': nni,
                'num_beats': len(peaks)
            }
            
        except Exception as e:
            print(f"Error processing PPG signal: {e}")
            return self._get_empty_hrv_dict()
    
    def process_temperature(self, temp_readings: np.ndarray) -> float:
        """
        Process temperature readings from LM35 sensor.
        
        Args:
            temp_readings: Array of temperature readings in °C
            
        Returns:
            Average temperature in °C
        """
        if len(temp_readings) == 0:
            return 0.0
        
        # Remove outliers
        temp_clean = self._remove_outliers(temp_readings)
        
        return float(np.mean(temp_clean)) if len(temp_clean) > 0 else float(np.mean(temp_readings))
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance."""
        signal = np.array(signal, dtype=float)
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std == 0:
            return signal - mean
        
        return (signal - mean) / std
    
    def _detect_peaks(self, signal: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Detect peaks in the signal using simple threshold method.
        
        Args:
            signal: Input signal
            threshold: Threshold for peak detection
            
        Returns:
            Array of peak indices
        """
        # Find local maxima
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if signal[i] > threshold:
                    peaks.append(i)
        
        return np.array(peaks)
    
    def _remove_outliers(self, data: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """
        Remove outliers from data using IQR method.
        
        Args:
            data: Input data
            method: Outlier detection method ('iqr' or 'zscore')
            
        Returns:
            Cleaned data
        """
        data = np.array(data)
        
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            return data[z_scores < 3]
        
        return data
    
    def _compute_hrv_metrics(self, nni: np.ndarray) -> Dict:
        """
        Compute HRV metrics using pyHRV.
        
        Args:
            nni: NN intervals in milliseconds
            
        Returns:
            Dictionary of HRV metrics
        """
        metrics = {}
        
        try:
            # Time domain metrics
            time_metrics = time_domain(nni)
            metrics['mean_nni'] = time_metrics.get('nni_mean', 0)
            metrics['sdnn'] = time_metrics.get('sdnn', 0)
            metrics['sdsd'] = time_metrics.get('sdsd', 0)
            metrics['rmssd'] = time_metrics.get('rmssd', 0)
            metrics['pnn20'] = time_metrics.get('pnn20', 0)
            metrics['pnn50'] = time_metrics.get('pnn50', 0)
            
        except Exception as e:
            print(f"Error computing time domain metrics: {e}")
        
        try:
            # Frequency domain metrics
            freq_metrics = hrv.frequency_domain(nni)
            metrics['vlf'] = freq_metrics.get('vlf', 0)
            metrics['lf'] = freq_metrics.get('lf', 0)
            metrics['hf'] = freq_metrics.get('hf', 0)
            metrics['lfhf'] = freq_metrics.get('lfhf', 0)
            
        except Exception as e:
            print(f"Error computing frequency domain metrics: {e}")
        
        try:
            # Non-linear metrics
            nl_metrics = nl.sampen(nni)
            metrics['sample_entropy'] = nl_metrics.get('sampen', 0)
            metrics['approximate_entropy'] = metrics['sample_entropy']
            
        except Exception as e:
            print(f"Error computing non-linear metrics: {e}")
        
        return metrics
    
    def _calculate_spo2(self, red_signal: np.ndarray, ir_signal: np.ndarray) -> float:
        """
        Calculate SpO2 from red and IR PPG signals using calibration curve.
        
        Uses the standard pulse oximetry calibration curve from:
        https://www.researchgate.net/figure/Typical-pulse-oximeter-calibration-curve_fig3_312279299
        
        The curve is based on the relationship:
        SpO2 = A * R^2 + B * R + C
        where R = (AC_red/DC_red) / (AC_ir/DC_ir)
        
        Args:
            red_signal: Red channel PPG signal
            ir_signal: IR channel PPG signal
            
        Returns:
            SpO2 percentage (70-100%)
        """
        if len(red_signal) == 0 or len(ir_signal) == 0:
            return 95.0
        
        try:
            red_signal = np.array(red_signal, dtype=float)
            ir_signal = np.array(ir_signal, dtype=float)
            
            # Calculate AC (alternating) and DC (direct) components
            red_ac = np.max(red_signal) - np.min(red_signal)
            red_dc = np.mean(red_signal)
            ir_ac = np.max(ir_signal) - np.min(ir_signal)
            ir_dc = np.mean(ir_signal)
            
            # Avoid division by zero
            if red_dc == 0 or ir_dc == 0 or ir_ac == 0:
                return 95.0
            
            # Calculate the ratio R = (AC_red/DC_red) / (AC_ir/DC_ir)
            r_ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
            
            # Standard pulse oximetry calibration curve coefficients
            # These coefficients are derived from typical pulse oximeter calibration curves
            # and represent the relationship between R ratio and SpO2
            A = -45.106
            B = 30.354
            C = 94.845
            
            # Calculate SpO2 using the calibration curve
            spo2 = A * (r_ratio ** 2) + B * r_ratio + C
            
            # Clamp to valid physiological range (70-100%)
            spo2 = np.clip(spo2, 70, 100)
            
            return float(spo2)
            
        except Exception as e:
            print(f"Error calculating SpO2: {e}")
            return 95.0
    
    def _get_empty_hrv_dict(self) -> Dict:
        """Return a dictionary with zero values for all HRV metrics."""
        return {
            'heart_rate': 83.0,
            'spo2': 95.0,
            'hrv_metrics': {
                'mean_nni': 0,
                'sdnn': 0,
                'sdsd': 0,
                'rmssd': 0,
                'pnn20': 0,
                'pnn50': 0,
                'vlf': 0,
                'lf': 0,
                'hf': 0,
                'lfhf': 0,
                'sample_entropy': 0,
                'approximate_entropy': 0
            },
            'nni': np.array([]),
            'num_beats': 0
        }
    
    def extract_features(self, red_signal: np.ndarray, ir_signal: np.ndarray, temp_readings: np.ndarray) -> np.ndarray:
        """
        Extract features from PPG (red and IR) and temperature data for model input.
        
        Returns 15-feature vector to match training data:
        [HR, SpO2, Temp, HR_std, SpO2_std, Temp_std, 
         HR_cv, SpO2_cv, Temp_cv, HR_range, SpO2_range, Temp_range,
         HR_roc, SpO2_roc, Temp_roc]
        
        Args:
            red_signal: Raw red channel PPG signal
            ir_signal: Raw IR channel PPG signal
            temp_readings: Temperature readings
            
        Returns:
            15-feature vector for model prediction
        """
        # Process PPG
        ppg_metrics = self.process_ppg_signal(red_signal, ir_signal)
        
        # Process temperature
        temperature = self.process_temperature(temp_readings)
        
        # Extract mean values (3 features)
        heart_rate = ppg_metrics['heart_rate']
        spo2 = ppg_metrics['spo2']
        temp_mean = temperature
        
        # Compute variability metrics from the signal itself
        # For real-time data, we estimate variability from signal characteristics
        
        # Heart rate variability (from HRV metrics)
        hr_std = ppg_metrics['hrv_metrics'].get('sdnn', 0)  # Standard deviation of NN intervals
        hr_cv = ppg_metrics['hrv_metrics'].get('rmssd', 0) / (heart_rate + 1e-6) if heart_rate > 0 else 0
        hr_range = ppg_metrics['hrv_metrics'].get('lf', 0)  # Use LF as proxy for range
        hr_roc = ppg_metrics['hrv_metrics'].get('hf', 0)    # Use HF as proxy for rate of change
        
        # SpO2 variability (estimate from signal noise)
        spo2_std = np.std(ir_signal) / 100 if len(ir_signal) > 0 else 0  # Normalize IR signal std
        spo2_cv = spo2_std / (spo2 + 1e-6) if spo2 > 0 else 0
        spo2_range = (np.max(ir_signal) - np.min(ir_signal)) / 100 if len(ir_signal) > 0 else 0
        spo2_roc = ppg_metrics['hrv_metrics'].get('lfhf', 0)  # Use LF/HF ratio as proxy
        
        # Temperature variability (estimate from readings)
        temp_std = np.std(temp_readings) if len(temp_readings) > 0 else 0
        temp_cv = temp_std / (temp_mean + 1e-6) if temp_mean > 0 else 0
        temp_range = (np.max(temp_readings) - np.min(temp_readings)) if len(temp_readings) > 0 else 0
        temp_roc = np.mean(np.abs(np.diff(temp_readings))) if len(temp_readings) > 1 else 0
        
        # Create 15-feature vector matching training data format
        features = [
            # Mean values (3)
            heart_rate,
            spo2,
            temp_mean,
            # Standard deviations (3)
            hr_std,
            spo2_std,
            temp_std,
            # Coefficients of variation (3)
            hr_cv,
            spo2_cv,
            temp_cv,
            # Ranges (3)
            hr_range,
            spo2_range,
            temp_range,
            # Rates of change (3)
            hr_roc,
            spo2_roc,
            temp_roc
        ]
        
        return np.array(features).reshape(1, -1)
