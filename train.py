#!/usr/bin/env python3
"""
Master training script for the delirium detection ensemble model.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import load_combined_data, generate_synthetic_data
from models.ensemble import DeliriumEnsemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    classification_report
)

from typing import Dict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train delirium detection ensemble model'
    )
    parser.add_argument(
        '--mimic_path', type=str, default=None,
        help='Path to MIMIC-IV dataset'
    )
    parser.add_argument(
        '--eicu_path', type=str, default=None,
        help='Path to eICU dataset'
    )
    parser.add_argument(
        '--output_dir', type=str, default='models/saved_models',
        help='Directory to save trained model'
    )
    parser.add_argument(
        '--test_size', type=float, default=0.2,
        help='Proportion of data to use for testing'
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--use_synthetic', action='store_true',
        help='Use synthetic data for testing'
    )
    parser.add_argument(
        '--synthetic_samples', type=int, default=1000,
        help='Number of synthetic samples to generate'
    )
    
    return parser.parse_args()


def load_data(args) -> tuple:
    """Load training data."""
    if args.use_synthetic:
        print("Generating synthetic data...")
        X, y = generate_synthetic_data(n_samples=args.synthetic_samples)
        print(f"Generated {len(X)} synthetic samples")
    else:
        if not args.mimic_path and not args.eicu_path:
            print("No data paths provided. Using synthetic data...")
            X, y = generate_synthetic_data(n_samples=args.synthetic_samples)
        else:
            print("Loading real data...")
            X, y = load_combined_data(
                mimic_path=args.mimic_path or '',
                eicu_path=args.eicu_path or ''
            )
    
    return X, y


def train_ensemble(X_train: np.ndarray, y_train: np.ndarray) -> DeliriumEnsemble:
    """Train the ensemble model."""
    print("\n" + "="*60)
    print("Training Ensemble Model")
    print("="*60)
    
    ensemble = DeliriumEnsemble(
        kmeans_params={'n_clusters': 3},
        ocsvm_params={'nu': 0.1, 'kernel': 'rbf'},
        gaussian_params={'contamination': 0.1},
        svm_params={'C': 1.0, 'kernel': 'rbf', 'class_weight': 'balanced'}
    )
    
    ensemble.fit(X_train, y_train)
    
    print("✓ Ensemble model trained successfully")
    return ensemble


def evaluate_model(model: DeliriumEnsemble, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate the model on test data."""
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
        'pr_auc': average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print metrics
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return metrics


def save_model(model: DeliriumEnsemble, output_dir: str):
    """Save the trained model."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'delirium_ensemble.joblib')
    
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    return model_path


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.random_state)
    
    try:
        # Load data
        print("="*60)
        print("Loading Data")
        print("="*60)
        X, y = load_data(args)
        
        print(f"Total samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Class distribution: {np.bincount(y.astype(int))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Train class distribution: {np.bincount(y_train.astype(int))}")
        print(f"Test class distribution: {np.bincount(y_test.astype(int))}")
        
        # Train model
        model = train_ensemble(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        model_path = save_model(model, args.output_dir)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
