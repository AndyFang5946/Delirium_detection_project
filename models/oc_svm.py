"""
One-Class SVM model for unsupervised anomaly detection.
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional


class OCSVMDetector:
    """One-Class SVM for anomaly detection."""
    
    def __init__(self, nu: float = 0.1, kernel: str = 'rbf', gamma: str = 'scale'):
        """
        Initialize the One-Class SVM model.
        
        Args:
            nu: An upper bound on the fraction of training errors and a lower bound 
                of the fraction of support vectors. Should be in the interval (0, 1].
            kernel: Kernel type for SVM.
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'OCSVMDetector':
        """
        Fit the One-Class SVM model on normal data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            self
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        Higher score = more normal (inlier).
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Anomaly scores (n_samples, 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Get decision function values (signed distance to the separating hyperplane)
        decision_scores = self.model.decision_function(X_scaled)
        
        # Convert to probability-like scores using sigmoid function
        # The decision_function returns + for inliers and - for outliers
        # We'll convert this to a probability using the logistic function
        scores = 1 / (1 + np.exp(-decision_scores))
        
        return scores.reshape(-1, 1)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters."""
        return {
            'nu': self.nu,
            'kernel': self.kernel,
            'gamma': self.gamma
        }
    
    def set_params(self, **params) -> 'OCSVMDetector':
        """Set model parameters."""
        for param, value in params.items():
            setattr(self, param, value)
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
        )
        self.is_fitted = False
        return self
