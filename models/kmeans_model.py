"""
K-means clustering model for unsupervised anomaly detection.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional


class KMeansDetector:
    """K-means clustering for anomaly detection."""
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initialize the K-means model.
        
        Args:
            n_clusters: Number of clusters (k)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.cluster_centers_ = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'KMeansDetector':
        """
        Fit the K-means model.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            self
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.cluster_centers_ = self.model.cluster_centers_
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores based on distance to cluster centers.
        Higher score = more normal (closer to cluster center).
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Anomaly scores (n_samples, 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Calculate distances to all cluster centers
        distances = np.array([
            np.linalg.norm(X_scaled - center, axis=1) 
            for center in self.cluster_centers_
        ])
        
        # Get minimum distance to any cluster center
        min_distances = np.min(distances, axis=0)
        
        # Convert to probability-like scores (lower distance = higher probability of being normal)
        # Using exponential decay to convert distances to [0,1] range
        max_dist = np.max(min_distances)
        if max_dist > 0:
            scores = np.exp(-min_distances / max_dist)
        else:
            scores = np.ones_like(min_distances)
        
        return scores.reshape(-1, 1)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters."""
        return {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'KMeansDetector':
        """Set model parameters."""
        for param, value in params.items():
            setattr(self, param, value)
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.is_fitted = False
        return self
