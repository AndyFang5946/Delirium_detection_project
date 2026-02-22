"""
Ensemble model combining K-means, One-Class SVM, Gaussian, and Binary SVM.
"""

import numpy as np
from typing import Dict, Optional
from .kmeans_model import KMeansDetector
from .oc_svm import OCSVMDetector
from .gaussian_model import GaussianDetector
from .binary_svm import BinarySVMClassifier


class DeliriumEnsemble:
    """Ensemble model for delirium detection."""
    
    def __init__(self, 
                 kmeans_params: Optional[Dict] = None,
                 ocsvm_params: Optional[Dict] = None,
                 gaussian_params: Optional[Dict] = None,
                 svm_params: Optional[Dict] = None,
                 weights: Optional[Dict] = None):
        """
        Initialize the ensemble model.
        
        Args:
            kmeans_params: Parameters for KMeansDetector
            ocsvm_params: Parameters for OCSVMDetector
            gaussian_params: Parameters for GaussianDetector
            svm_params: Parameters for BinarySVMClassifier
            weights: Dictionary with weights for each model's prediction
        """
        # Initialize models with default or provided parameters
        self.kmeans = KMeansDetector(**(kmeans_params or {'n_clusters': 3}))
        self.ocsvm = OCSVMDetector(**(ocsvm_params or {}))
        self.gaussian = GaussianDetector(**(gaussian_params or {}))
        self.svm = BinarySVMClassifier(**(svm_params or {}))
        
        # Set default weights if not provided
        self.weights = weights or {
            'kmeans': 0.15,
            'ocsvm': 0.15,
            'gaussian': 0.20,
            'svm': 0.50  # Higher weight for supervised model
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DeliriumEnsemble':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,). 0 = normal, 1 = delirium
            
        Returns:
            self
        """
        if y is None:
            raise ValueError("Target labels (y) are required for ensemble training")
        
        # Fit unsupervised models on normal data (y=0)
        X_normal = X[y == 0]
        
        if len(X_normal) < 10:
            raise ValueError("Not enough normal samples for training unsupervised models")
        
        print(f"Training unsupervised models on {len(X_normal)} normal samples...")
        self.kmeans.fit(X_normal)
        self.ocsvm.fit(X_normal)
        self.gaussian.fit(X_normal)
        
        # Fit supervised model on all data
        print(f"Training supervised SVM on {len(X)} total samples...")
        self.svm.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of delirium using ensemble.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Probability scores (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all models
        kmeans_scores = self.kmeans.predict_proba(X)
        ocsvm_scores = self.ocsvm.predict_proba(X)
        gaussian_scores = self.gaussian.predict_proba(X)
        
        # For the unsupervised models, invert the scores to get anomaly scores
        # (higher score = more likely to be an anomaly/delirium)
        kmeans_anomaly = 1 - kmeans_scores
        ocsvm_anomaly = 1 - ocsvm_scores
        gaussian_anomaly = 1 - gaussian_scores
        
        # Get supervised model predictions
        svm_scores = self.svm.predict_proba(X)
        
        # Combine predictions using weighted average
        ensemble_scores = (
            self.weights['kmeans'] * kmeans_anomaly +
            self.weights['ocsvm'] * ocsvm_anomaly +
            self.weights['gaussian'] * gaussian_anomaly +
            self.weights['svm'] * svm_scores
        )
        
        return ensemble_scores.flatten()
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features (n_samples, n_features)
            threshold: Decision threshold
            
        Returns:
            Predicted class labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters."""
        return {
            'kmeans_params': self.kmeans.get_params(),
            'ocsvm_params': self.ocsvm.get_params(),
            'gaussian_params': self.gaussian.get_params(),
            'svm_params': self.svm.get_params(),
            'weights': self.weights
        }
    
    def set_params(self, **params) -> 'DeliriumEnsemble':
        """Set model parameters."""
        # Extract weights if provided
        if 'weights' in params:
            self.weights = params.pop('weights')
            # Normalize weights
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Update model parameters
        for param, value in params.items():
            if param.endswith('_params'):
                model_name = param.split('_params')[0]
                getattr(self, model_name).set_params(**value)
        
        return self
