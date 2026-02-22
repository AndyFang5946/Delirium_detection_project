"""
Binary SVM classifier for supervised delirium detection.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from typing import Dict, Optional, Union


class BinarySVMClassifier:
    """Binary SVM classifier for delirium detection."""
    
    def __init__(self, 
                 C: float = 1.0, 
                 kernel: str = 'rbf',
                 gamma: str = 'scale',
                 class_weight: Optional[Union[Dict, str]] = 'balanced'):
        """
        Initialize the binary SVM classifier.
        
        Args:
            C: Regularization parameter.
            kernel: Kernel type.
            gamma: Kernel coefficient.
            class_weight: Weights associated with classes.
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        
        # Use SVC with probability=True for direct probability estimation
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.classes_ = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BinarySVMClassifier':
        """
        Fit the binary SVM model.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            self
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Probability of positive class (delirium) (n_samples, 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities for both classes
        proba = self.model.predict_proba(X_scaled)
        
        # Return probability of positive class (delirium)
        # proba[:, 1] is the probability of class 1 (delirium)
        return proba[:, 1].reshape(-1, 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters."""
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'class_weight': self.class_weight
        }
    
    def set_params(self, **params) -> 'BinarySVMClassifier':
        """Set model parameters."""
        for param, value in params.items():
            setattr(self, param, value)
        
        # Reinitialize the model with new parameters
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            class_weight=self.class_weight,
            probability=True,
            random_state=42
        )
        
        self.is_fitted = False
        return self
