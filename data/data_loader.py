"""
Data loader for MIMIC-IV and eICU datasets.
Extracts heart rate, blood oxygen, and temperature data.
Computes variability metrics (HRV) from measurement timestamps.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import wfdb
from datetime import datetime


class VariabilityCalculator:
    """Calculate variability metrics from time-series vital signs."""
    
    @staticmethod
    def calculate_variability(values: np.ndarray, timestamps: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate variability metrics from a series of measurements.
        
        Args:
            values: Array of vital sign values
            timestamps: Array of measurement timestamps (optional)
            
        Returns:
            Dictionary with variability metrics
        """
        if len(values) < 2:
            return {
                'mean': float(np.mean(values)) if len(values) > 0 else 0.0,
                'std': 0.0,
                'cv': 0.0,  # Coefficient of variation
                'range': 0.0,
                'rate_of_change': 0.0
            }
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Coefficient of variation (normalized std)
        cv = (std_val / mean_val) if mean_val != 0 else 0.0
        
        # Range
        range_val = np.max(values) - np.min(values)
        
        # Rate of change (if timestamps available)
        rate_of_change = 0.0
        if timestamps is not None and len(timestamps) > 1:
            # Calculate time differences in minutes
            time_diffs = np.diff(timestamps) / 60.0  # Convert seconds to minutes
            value_diffs = np.diff(values)
            
            # Avoid division by zero
            valid_idx = time_diffs > 0
            if np.any(valid_idx):
                rates = value_diffs[valid_idx] / time_diffs[valid_idx]
                rate_of_change = np.mean(np.abs(rates))
        
        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'cv': float(cv),
            'range': float(range_val),
            'rate_of_change': float(rate_of_change)
        }


class MIMICDataLoader:
    """Load and preprocess data from MIMIC-IV dataset."""
    
    def __init__(self, mimic_path: str):
        """
        Initialize the MIMIC data loader.
        
        Args:
            mimic_path: Path to the MIMIC-IV dataset root directory
        """
        self.mimic_path = mimic_path
        
    def load_vitals(self) -> pd.DataFrame:
        """
        Load vital signs from MIMIC-IV with timestamps.
        Extracts: heart_rate, spo2 (blood oxygen), temperature
        Computes variability metrics from measurement timestamps.
        
        Returns:
            DataFrame with columns: subject_id, hadm_id, heart_rate, spo2, temperature,
                                   hr_std, spo2_std, temp_std, hr_cv, spo2_cv, temp_cv,
                                   hr_range, spo2_range, temp_range, hr_roc, spo2_roc, temp_roc
        """
        vitals_path = os.path.join(self.mimic_path, 'icu', 'chartevents.csv')
        
        if not os.path.exists(vitals_path):
            raise FileNotFoundError(f"Vitals file not found at {vitals_path}")
        
        # Load chartevents with relevant columns including timestamps
        # itemid: 220045 = heart rate, 220277 = spo2, 223761 = temperature
        df = pd.read_csv(vitals_path, usecols=['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum'])
        
        # Filter for relevant vital signs
        vitals_itemids = {
            220045: 'heart_rate',      # Heart Rate
            220277: 'spo2',            # SpO2
            223761: 'temperature'      # Temperature (Celsius)
        }
        
        df = df[df['itemid'].isin(vitals_itemids.keys())].copy()
        df['vital_type'] = df['itemid'].map(vitals_itemids)
        
        # Convert charttime to datetime and then to timestamp (seconds)
        df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
        df = df.dropna(subset=['charttime'])
        df['timestamp'] = df['charttime'].astype(np.int64) / 1e9  # Convert to seconds
        
        # Group by patient and vital type to compute statistics
        vitals_list = []
        
        for (subject_id, hadm_id), group in df.groupby(['subject_id', 'hadm_id']):
            vital_data = {'subject_id': subject_id, 'hadm_id': hadm_id}
            
            for vital_type in vitals_itemids.values():
                vital_group = group[group['vital_type'] == vital_type]
                
                if len(vital_group) == 0:
                    # No data for this vital
                    vital_data[vital_type] = np.nan
                    vital_data[f'{vital_type}_std'] = 0.0
                    vital_data[f'{vital_type}_cv'] = 0.0
                    vital_data[f'{vital_type}_range'] = 0.0
                    vital_data[f'{vital_type}_roc'] = 0.0
                else:
                    values = vital_group['valuenum'].values
                    timestamps = vital_group['timestamp'].values
                    
                    # Calculate variability
                    var_metrics = VariabilityCalculator.calculate_variability(values, timestamps)
                    
                    vital_data[vital_type] = var_metrics['mean']
                    vital_data[f'{vital_type}_std'] = var_metrics['std']
                    vital_data[f'{vital_type}_cv'] = var_metrics['cv']
                    vital_data[f'{vital_type}_range'] = var_metrics['range']
                    vital_data[f'{vital_type}_roc'] = var_metrics['rate_of_change']
            
            vitals_list.append(vital_data)
        
        vitals = pd.DataFrame(vitals_list)
        return vitals
    
    def load_delirium_labels(self) -> pd.DataFrame:
        """
        Load delirium labels from MIMIC-IV using CAM-ICU codes.
        CAM-ICU is the gold standard for delirium assessment in ICU.
        
        Returns:
            DataFrame with columns: subject_id, hadm_id, has_delirium
        """
        chartevents_path = os.path.join(self.mimic_path, 'icu', 'chartevents.csv')
        
        if not os.path.exists(chartevents_path):
            # Fallback to using ICD codes if chartevents not available
            return self._load_delirium_labels_icd()
        
        try:
            # CAM-ICU itemids in MIMIC-IV
            # itemid 230331: CAM-ICU result (1=positive/delirium, 0=negative)
            # itemid 226993: Delirium assessment
            # itemid 227012: Delirium status
            cam_itemids = [230331, 226993, 227012]
            
            df = pd.read_csv(chartevents_path, 
                            usecols=['subject_id', 'hadm_id', 'itemid', 'value', 'valuenum'])
            
            # Filter for CAM-ICU assessments
            df = df[df['itemid'].isin(cam_itemids)].copy()
            
            if len(df) == 0:
                # No CAM-ICU data, fallback to ICD codes
                return self._load_delirium_labels_icd()
            
            # Parse CAM-ICU results
            # Positive values indicate delirium
            df['has_delirium'] = 0
            
            # For itemid 230331: value field contains 'Positive' or 'Negative'
            if 230331 in df['itemid'].values:
                cam_230331 = df[df['itemid'] == 230331].copy()
                cam_230331['has_delirium'] = cam_230331['value'].str.lower().str.contains('positive', na=False).astype(int)
                df = pd.concat([df[df['itemid'] != 230331], cam_230331], ignore_index=True)
            
            # For itemid 226993 and 227012: valuenum field (1=positive, 0=negative)
            for itemid in [226993, 227012]:
                if itemid in df['itemid'].values:
                    mask = df['itemid'] == itemid
                    df.loc[mask, 'has_delirium'] = (df.loc[mask, 'valuenum'] == 1).astype(int)
            
            # Get unique patients with delirium status (max = any positive CAM-ICU)
            labels = df.groupby(['subject_id', 'hadm_id'])['has_delirium'].max().reset_index()
            
            return labels
            
        except Exception as e:
            print(f"Error loading CAM-ICU data: {e}. Falling back to ICD codes.")
            return self._load_delirium_labels_icd()
    
    def _load_delirium_labels_icd(self) -> pd.DataFrame:
        """
        Fallback method: Load delirium labels from ICD-10 codes.
        Less reliable than CAM-ICU but available when CAM data is missing.
        
        Returns:
            DataFrame with columns: subject_id, hadm_id, has_delirium
        """
        diagnoses_path = os.path.join(self.mimic_path, 'hosp', 'diagnoses_icd.csv')
        
        if not os.path.exists(diagnoses_path):
            raise FileNotFoundError(f"Diagnoses file not found at {diagnoses_path}")
        
        df = pd.read_csv(diagnoses_path, usecols=['subject_id', 'hadm_id', 'icd_code'])
        
        # Delirium ICD-10 codes
        delirium_codes = ['F05', 'F051', 'F052', 'F053']
        
        # Mark patients with delirium
        df['has_delirium'] = df['icd_code'].str.startswith(tuple(delirium_codes)).astype(int)
        
        # Get unique patients with delirium status
        labels = df.groupby(['subject_id', 'hadm_id'])['has_delirium'].max().reset_index()
        
        return labels
    
    def load_and_merge(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load vitals and labels, merge them, and return features and target.
        
        Returns:
            Tuple of (features array, labels array)
        """
        vitals = self.load_vitals()
        labels = self.load_delirium_labels()
        
        # Merge vitals and labels
        data = vitals.merge(labels, on=['subject_id', 'hadm_id'], how='inner')
        
        # Remove rows with missing values in vital signs
        data = data.dropna(subset=['heart_rate', 'spo2', 'temperature'])
        
        # Extract features with variability metrics
        # Feature vector: [HR, SpO2, Temp, HR_std, SpO2_std, Temp_std, 
        #                  HR_cv, SpO2_cv, Temp_cv, HR_range, SpO2_range, Temp_range,
        #                  HR_roc, SpO2_roc, Temp_roc]
        feature_cols = [
            'heart_rate', 'spo2', 'temperature',
            'heart_rate_std', 'spo2_std', 'temperature_std',
            'heart_rate_cv', 'spo2_cv', 'temperature_cv',
            'heart_rate_range', 'spo2_range', 'temperature_range',
            'heart_rate_roc', 'spo2_roc', 'temperature_roc'
        ]
        
        # Ensure all required columns exist
        for col in feature_cols:
            if col not in data.columns:
                data[col] = 0.0
        
        X = data[feature_cols].values
        y = data['has_delirium'].values
        
        return X, y


class eICUDataLoader:
    """Load and preprocess data from eICU dataset."""
    
    def __init__(self, eicu_path: str):
        """
        Initialize the eICU data loader.
        
        Args:
            eicu_path: Path to the eICU dataset root directory
        """
        self.eicu_path = eicu_path
        
    def load_vitals(self) -> pd.DataFrame:
        """
        Load vital signs from eICU with timestamps.
        Extracts: heart_rate, spo2 (blood oxygen), temperature
        Computes variability metrics from measurement timestamps.
        
        Returns:
            DataFrame with columns: patientunitstayid, heart_rate, spo2, temperature,
                                   hr_std, spo2_std, temp_std, hr_cv, spo2_cv, temp_cv,
                                   hr_range, spo2_range, temp_range, hr_roc, spo2_roc, temp_roc
        """
        vitals_path = os.path.join(self.eicu_path, 'vitalPeriodic.csv')
        
        if not os.path.exists(vitals_path):
            raise FileNotFoundError(f"Vitals file not found at {vitals_path}")
        
        df = pd.read_csv(vitals_path, usecols=[
            'patientunitstayid', 'observationoffset', 'heartrate', 'sao2', 'temperature'
        ])
        
        # Rename columns for consistency
        df = df.rename(columns={
            'heartrate': 'heart_rate',
            'sao2': 'spo2',
            'temperature': 'temperature'
        })
        
        # Convert observationoffset to timestamp (minutes since admission, convert to seconds)
        df['timestamp'] = df['observationoffset'] * 60.0  # Convert minutes to seconds
        
        # Group by patient to compute statistics
        vitals_list = []
        
        for patient_id, group in df.groupby('patientunitstayid'):
            vital_data = {'patientunitstayid': patient_id}
            
            for vital_type in ['heart_rate', 'spo2', 'temperature']:
                vital_values = group[vital_type].dropna().values
                timestamps = group.loc[group[vital_type].notna(), 'timestamp'].values
                
                if len(vital_values) == 0:
                    # No data for this vital
                    vital_data[vital_type] = np.nan
                    vital_data[f'{vital_type}_std'] = 0.0
                    vital_data[f'{vital_type}_cv'] = 0.0
                    vital_data[f'{vital_type}_range'] = 0.0
                    vital_data[f'{vital_type}_roc'] = 0.0
                else:
                    # Calculate variability
                    var_metrics = VariabilityCalculator.calculate_variability(vital_values, timestamps)
                    
                    vital_data[vital_type] = var_metrics['mean']
                    vital_data[f'{vital_type}_std'] = var_metrics['std']
                    vital_data[f'{vital_type}_cv'] = var_metrics['cv']
                    vital_data[f'{vital_type}_range'] = var_metrics['range']
                    vital_data[f'{vital_type}_roc'] = var_metrics['rate_of_change']
            
            vitals_list.append(vital_data)
        
        vitals = pd.DataFrame(vitals_list)
        return vitals
    
    def load_delirium_labels(self) -> pd.DataFrame:
        """
        Load delirium labels from eICU using CAM-ICU assessments.
        eICU stores CAM-ICU results in the nurseCharting table.
        
        Returns:
            DataFrame with columns: patientunitstayid, has_delirium
        """
        # Try to load from nurseCharting table (CAM-ICU assessments)
        nurse_charting_path = os.path.join(self.eicu_path, 'nurseCharting.csv')
        
        if os.path.exists(nurse_charting_path):
            try:
                df = pd.read_csv(nurse_charting_path, 
                                usecols=['patientunitstayid', 'nursingchartcelltypevalname', 'nursingchartvalue'])
                
                # CAM-ICU related nursing chart entries
                # Look for entries containing 'CAM' or 'delirium' in the cell type
                cam_mask = df['nursingchartcelltypevalname'].str.lower().str.contains('cam|delirium', na=False)
                df = df[cam_mask].copy()
                
                if len(df) > 0:
                    # Parse CAM-ICU results
                    # Positive values indicate delirium
                    df['has_delirium'] = df['nursingchartvalue'].str.lower().str.contains(
                        'positive|yes|delirium', na=False
                    ).astype(int)
                    
                    # Get unique patients with delirium status
                    labels = df.groupby('patientunitstayid')['has_delirium'].max().reset_index()
                    
                    return labels
            except Exception as e:
                print(f"Error loading CAM-ICU from nurseCharting: {e}")
        
        # Fallback: Use diagnosis table with keyword matching
        return self._load_delirium_labels_diagnosis()
    
    def _load_delirium_labels_diagnosis(self) -> pd.DataFrame:
        """
        Fallback method: Load delirium labels from diagnosis table.
        Less reliable than CAM-ICU but available as backup.
        
        Returns:
            DataFrame with columns: patientunitstayid, has_delirium
        """
        diagnosis_path = os.path.join(self.eicu_path, 'diagnosis.csv')
        
        if not os.path.exists(diagnosis_path):
            raise FileNotFoundError(f"Diagnosis file not found at {diagnosis_path}")
        
        df = pd.read_csv(diagnosis_path, usecols=['patientunitstayid', 'diagnosisstring'])
        
        # Search for delirium-related diagnoses
        # Using ICD-9 and ICD-10 codes for delirium
        delirium_patterns = [
            'delirium',
            'confusion',
            'altered mental',
            'encephalopathy',
            'acute brain',
            'mental status change'
        ]
        
        df['has_delirium'] = df['diagnosisstring'].str.lower().str.contains(
            '|'.join(delirium_patterns), na=False
        ).astype(int)
        
        # Get unique patients with delirium status
        labels = df.groupby('patientunitstayid')['has_delirium'].max().reset_index()
        
        return labels
    
    def load_and_merge(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load vitals and labels, merge them, and return features and target.
        
        Returns:
            Tuple of (features array, labels array)
        """
        vitals = self.load_vitals()
        labels = self.load_delirium_labels()
        
        # Merge vitals and labels
        data = vitals.merge(labels, on='patientunitstayid', how='inner')
        
        # Remove rows with missing values in vital signs
        data = data.dropna(subset=['heart_rate', 'spo2', 'temperature'])
        
        # Extract features with variability metrics
        # Feature vector: [HR, SpO2, Temp, HR_std, SpO2_std, Temp_std, 
        #                  HR_cv, SpO2_cv, Temp_cv, HR_range, SpO2_range, Temp_range,
        #                  HR_roc, SpO2_roc, Temp_roc]
        feature_cols = [
            'heart_rate', 'spo2', 'temperature',
            'heart_rate_std', 'spo2_std', 'temperature_std',
            'heart_rate_cv', 'spo2_cv', 'temperature_cv',
            'heart_rate_range', 'spo2_range', 'temperature_range',
            'heart_rate_roc', 'spo2_roc', 'temperature_roc'
        ]
        
        # Ensure all required columns exist
        for col in feature_cols:
            if col not in data.columns:
                data[col] = 0.0
        
        X = data[feature_cols].values
        y = data['has_delirium'].values
        
        return X, y


def load_combined_data(mimic_path: str, eicu_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine data from both MIMIC-IV and eICU datasets.
    
    Args:
        mimic_path: Path to MIMIC-IV dataset
        eicu_path: Path to eICU dataset
        
    Returns:
        Tuple of (combined features array, combined labels array)
    """
    X_list = []
    y_list = []
    
    # Load MIMIC data
    if os.path.exists(mimic_path):
        try:
            mimic_loader = MIMICDataLoader(mimic_path)
            X_mimic, y_mimic = mimic_loader.load_and_merge()
            X_list.append(X_mimic)
            y_list.append(y_mimic)
            print(f"Loaded {len(X_mimic)} samples from MIMIC-IV")
        except Exception as e:
            print(f"Error loading MIMIC data: {e}")
    
    # Load eICU data
    if os.path.exists(eicu_path):
        try:
            eicu_loader = eICUDataLoader(eicu_path)
            X_eicu, y_eicu = eicu_loader.load_and_merge()
            X_list.append(X_eicu)
            y_list.append(y_eicu)
            print(f"Loaded {len(X_eicu)} samples from eICU")
        except Exception as e:
            print(f"Error loading eICU data: {e}")
    
    # Combine datasets
    if X_list:
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y
    else:
        raise ValueError("No data could be loaded from the provided paths")


def generate_synthetic_data(n_samples: int = 1000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing purposes.
    Includes variability metrics computed from timestamps.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed
        
    Returns:
        Tuple of (features array, labels array)
    """
    np.random.seed(random_state)
    
    # Generate synthetic vitals (mean values)
    # Normal ranges: HR 60-100, SpO2 95-100, Temp 36.5-37.5
    heart_rate = np.random.normal(75, 12, n_samples).clip(40, 180)
    spo2 = np.random.normal(97, 2, n_samples).clip(70, 100)
    temperature = np.random.normal(36.8, 0.5, n_samples).clip(35, 42)
    
    # Generate variability metrics (std, cv, range, rate of change)
    # Healthy individuals have lower variability
    hr_std = np.random.normal(5, 2, n_samples).clip(0, 20)
    spo2_std = np.random.normal(1, 0.5, n_samples).clip(0, 5)
    temp_std = np.random.normal(0.2, 0.1, n_samples).clip(0, 1)
    
    # Coefficient of variation
    hr_cv = hr_std / (heart_rate + 1e-6)
    spo2_cv = spo2_std / (spo2 + 1e-6)
    temp_cv = temp_std / (temperature + 1e-6)
    
    # Range
    hr_range = np.random.normal(15, 5, n_samples).clip(0, 50)
    spo2_range = np.random.normal(3, 1, n_samples).clip(0, 10)
    temp_range = np.random.normal(0.5, 0.2, n_samples).clip(0, 2)
    
    # Rate of change (per minute)
    hr_roc = np.random.normal(0.5, 0.3, n_samples).clip(0, 2)
    spo2_roc = np.random.normal(0.1, 0.05, n_samples).clip(0, 0.5)
    temp_roc = np.random.normal(0.05, 0.02, n_samples).clip(0, 0.2)
    
    # Create features: [HR, SpO2, Temp, HR_std, SpO2_std, Temp_std,
    #                   HR_cv, SpO2_cv, Temp_cv, HR_range, SpO2_range, Temp_range,
    #                   HR_roc, SpO2_roc, Temp_roc]
    X = np.column_stack((
        heart_rate, spo2, temperature,
        hr_std, spo2_std, temp_std,
        hr_cv, spo2_cv, temp_cv,
        hr_range, spo2_range, temp_range,
        hr_roc, spo2_roc, temp_roc
    ))
    
    # Generate labels (5% prevalence of delirium)
    # Patients with delirium tend to have higher variability
    y = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Add noise to variability for delirium patients
    delirium_mask = y == 1
    X[delirium_mask, 3:] *= np.random.uniform(1.5, 2.5, (np.sum(delirium_mask), 12))
    
    return X, y
