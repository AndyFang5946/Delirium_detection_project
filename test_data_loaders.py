#!/usr/bin/env python3
"""
Test script to verify data loaders can access datasets and extract labels.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import MIMICDataLoader, eICUDataLoader, generate_synthetic_data


def test_mimic_loader():
    """Test MIMIC-IV data loader."""
    print("\n" + "="*60)
    print("Testing MIMIC-IV Data Loader")
    print("="*60)
    
    mimic_path = "/Users/feng/Desktop/ship/datasets/mimic-iv-clinical-database-demo-2.2"
    
    if not os.path.exists(mimic_path):
        print(f"❌ MIMIC-IV dataset not found at {mimic_path}")
        return False
    
    print(f"✓ Dataset found at {mimic_path}")
    
    try:
        loader = MIMICDataLoader(mimic_path)
        
        # Test vitals loading
        print("\n1. Loading vitals...")
        vitals = loader.load_vitals()
        print(f"   ✓ Loaded {len(vitals)} patient records")
        print(f"   ✓ Columns: {list(vitals.columns)}")
        print(f"   ✓ Sample:\n{vitals.head()}")
        
        # Test labels loading
        print("\n2. Loading delirium labels...")
        labels = loader.load_delirium_labels()
        print(f"   ✓ Loaded {len(labels)} patient records")
        print(f"   ✓ Columns: {list(labels.columns)}")
        delirium_count = labels['has_delirium'].sum()
        print(f"   ✓ Delirium cases: {delirium_count}/{len(labels)} ({100*delirium_count/len(labels):.1f}%)")
        print(f"   ✓ Sample:\n{labels.head()}")
        
        # Test merged data
        print("\n3. Loading and merging data...")
        X, y = loader.load_and_merge()
        print(f"   ✓ Features shape: {X.shape}")
        print(f"   ✓ Labels shape: {y.shape}")
        print(f"   ✓ Feature columns: 15 (3 mean + 3 std + 3 cv + 3 range + 3 roc)")
        print(f"   ✓ Delirium prevalence: {y.sum()}/{len(y)} ({100*y.sum()/len(y):.1f}%)")
        print(f"   ✓ Sample features:\n{X[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eicu_loader():
    """Test eICU data loader."""
    print("\n" + "="*60)
    print("Testing eICU Data Loader")
    print("="*60)
    
    eicu_path = "/Users/feng/Desktop/ship/datasets/eicu-collaborative-research-database-demo-2.0.1"
    
    if not os.path.exists(eicu_path):
        print(f"❌ eICU dataset not found at {eicu_path}")
        return False
    
    print(f"✓ Dataset found at {eicu_path}")
    
    try:
        loader = eICUDataLoader(eicu_path)
        
        # Test vitals loading
        print("\n1. Loading vitals...")
        vitals = loader.load_vitals()
        print(f"   ✓ Loaded {len(vitals)} patient records")
        print(f"   ✓ Columns: {list(vitals.columns)}")
        print(f"   ✓ Sample:\n{vitals.head()}")
        
        # Test labels loading
        print("\n2. Loading delirium labels...")
        labels = loader.load_delirium_labels()
        print(f"   ✓ Loaded {len(labels)} patient records")
        print(f"   ✓ Columns: {list(labels.columns)}")
        delirium_count = labels['has_delirium'].sum()
        print(f"   ✓ Delirium cases: {delirium_count}/{len(labels)} ({100*delirium_count/len(labels):.1f}%)")
        print(f"   ✓ Sample:\n{labels.head()}")
        
        # Test merged data
        print("\n3. Loading and merging data...")
        X, y = loader.load_and_merge()
        print(f"   ✓ Features shape: {X.shape}")
        print(f"   ✓ Labels shape: {y.shape}")
        print(f"   ✓ Feature columns: 15 (3 mean + 3 std + 3 cv + 3 range + 3 roc)")
        print(f"   ✓ Delirium prevalence: {y.sum()}/{len(y)} ({100*y.sum()/len(y):.1f}%)")
        print(f"   ✓ Sample features:\n{X[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthetic_data():
    """Test synthetic data generation."""
    print("\n" + "="*60)
    print("Testing Synthetic Data Generation")
    print("="*60)
    
    try:
        print("\n1. Generating synthetic data...")
        X, y = generate_synthetic_data(n_samples=100)
        
        print(f"   ✓ Generated {len(X)} samples")
        print(f"   ✓ Features shape: {X.shape}")
        print(f"   ✓ Labels shape: {y.shape}")
        print(f"   ✓ Feature columns: 15")
        print(f"   ✓ Delirium prevalence: {y.sum()}/{len(y)} ({100*y.sum()/len(y):.1f}%)")
        print(f"   ✓ Sample features:\n{X[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Data Loader Verification Tests")
    print("="*60)
    
    results = {
        'MIMIC-IV': test_mimic_loader(),
        'eICU': test_eicu_loader(),
        'Synthetic': test_synthetic_data()
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nYou can now:")
        print("  1. Train models: python train.py")
        print("  2. Start webapp: python -m webapp.app")
        print("  3. Run sensor reader: python arduino/sensor_reader.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
