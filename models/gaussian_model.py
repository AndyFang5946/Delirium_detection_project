"""
Multivariate Gaussian model for unsupervised anomaly detection.
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional


class GaussianDetector:
    """Multivariate Gaussian for anomaly detection using robust covariance estimation."""
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize the Gaussian model.
        
        Args:
            contamination: The proportion of outliers in the data set.
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.covariance_ = None
        self.location_ = None
        self.mvn = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'GaussianDetector':
        """
        Fit the Gaussian model on normal data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            self
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Use Minimum Covariance Determinant for robust estimation
        support_fraction = max(0.5, 1 - self.contamination)
        robust_cov = MinCovDet(support_fraction=support_fraction, random_state=42)
        robust_cov.fit(X_scaled)
        
        self.location_ = robust_cov.location_
        self.covariance_ = robust_cov.covariance_
        
        # Create multivariate normal distribution
        self.mvn = multivariate_normal(
            mean=self.location_,
            cov=self.covariance_,
            allow_singular=True
        )
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute probability density scores.
        Higher score = more normal (higher probability density).
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Probability density scores (n_samples, 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Calculate log probability density
        log_probs = self.mvn.logpdf(X_scaled)
        
        # Convert to probability-like scores (higher density = more normal)
        # Normalize to [0,1] range
        min_log_prob = np.min(log_probs)
        max_log_prob = np.max(log_probs)
        
        if max_log_prob > min_log_prob:
            scores = (log_probs - min_log_prob) / (max_log_prob - min_log_prob)
        else:
            scores = np.ones_like(log_probs)
        
        return scores.reshape(-1, 1)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters."""
        return {'contamination': self.contamination}
    
    def set_params(self, **params) -> 'GaussianDetector':
        """Set model parameters."""
        for param, value in params.items():
            setattr(self, param, value)
        self.is_fitted = False
        return self
