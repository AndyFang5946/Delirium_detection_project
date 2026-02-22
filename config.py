"""
Configuration file for the delirium detection system.
Centralized settings for models, data, and application parameters.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Dataset paths
MIMIC_PATH = os.environ.get('MIMIC_PATH', '')
EICU_PATH = os.environ.get('EICU_PATH', '')

# Synthetic data parameters
SYNTHETIC_SAMPLES = 1000
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.2
STRATIFY = True

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# K-Means parameters
KMEANS_PARAMS = {
    'n_clusters': 3,
    'random_state': RANDOM_STATE
}

# One-Class SVM parameters
OCSVM_PARAMS = {
    'nu': 0.1,
    'kernel': 'rbf',
    'gamma': 'scale'
}

# Gaussian model parameters
GAUSSIAN_PARAMS = {
    'contamination': 0.1
}

# Binary SVM parameters
BINARY_SVM_PARAMS = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'class_weight': 'balanced'
}

# Ensemble weights (will be normalized)
ENSEMBLE_WEIGHTS = {
    'kmeans': 0.15,
    'ocsvm': 0.15,
    'gaussian': 0.20,
    'svm': 0.50
}

# ============================================================================
# SENSOR CONFIGURATION
# ============================================================================

# Sampling rate (Hz)
SAMPLING_RATE = 100

# PPG buffer size
PPG_BUFFER_SIZE = 100

# Temperature buffer size
TEMP_BUFFER_SIZE = 100

# Arduino serial configuration
ARDUINO_BAUD_RATE = 115200
ARDUINO_TIMEOUT = 1.0

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# Feature extraction
FEATURES = [
    'heart_rate',
    'rmssd',
    'sdnn',
    'lf',
    'hf',
    'lfhf',
    'temperature'
]

# Outlier detection method ('iqr' or 'zscore')
OUTLIER_METHOD = 'iqr'

# ============================================================================
# WEBAPP CONFIGURATION
# ============================================================================

# Flask app settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Patient data storage
MAX_PATIENT_HISTORY = 100

# Risk thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.7,
    'high': 1.0
}

# ============================================================================
# MODEL PATHS
# ============================================================================

# Model save directory
MODELS_DIR = PROJECT_ROOT / 'models' / 'saved_models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model file path
MODEL_PATH = MODELS_DIR / 'delirium_ensemble.joblib'

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_level(score: float) -> str:
    """Get risk level from score."""
    if score > RISK_THRESHOLDS['medium']:
        return 'HIGH'
    elif score > RISK_THRESHOLDS['low']:
        return 'MEDIUM'
    else:
        return 'LOW'


def get_model_config() -> dict:
    """Get complete model configuration."""
    return {
        'kmeans_params': KMEANS_PARAMS,
        'ocsvm_params': OCSVM_PARAMS,
        'gaussian_params': GAUSSIAN_PARAMS,
        'svm_params': BINARY_SVM_PARAMS,
        'weights': ENSEMBLE_WEIGHTS
    }


def get_training_config() -> dict:
    """Get training configuration."""
    return {
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'stratify': STRATIFY,
        'synthetic_samples': SYNTHETIC_SAMPLES
    }


def get_sensor_config() -> dict:
    """Get sensor configuration."""
    return {
        'sampling_rate': SAMPLING_RATE,
        'ppg_buffer_size': PPG_BUFFER_SIZE,
        'temp_buffer_size': TEMP_BUFFER_SIZE,
        'baud_rate': ARDUINO_BAUD_RATE,
        'timeout': ARDUINO_TIMEOUT
    }
