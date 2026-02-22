#!/usr/bin/env python3
"""
Test script to populate the webapp with synthetic patient data.
Simulates multiple patients with varying delirium risk levels.
"""

import requests
import numpy as np
import time
from datetime import datetime
from typing import Dict, List


class SyntheticPatientGenerator:
    """Generate synthetic patient data for testing."""
    
    def __init__(self, webapp_url: str = "http://localhost:5000"):
        """
        Initialize the generator.
        
        Args:
            webapp_url: Base URL of the web application
        """
        self.webapp_url = webapp_url
        self.update_endpoint = f"{webapp_url}/api/patients"
    
    def generate_patient_data(self, patient_id: str, risk_level: str) -> Dict:
        """
        Generate synthetic patient data.
        
        Args:
            patient_id: Patient identifier
            risk_level: "low", "medium", or "high"
            
        Returns:
            Dictionary with patient data
        """
        # Base values
        if risk_level == "low":
            hr = np.random.normal(75, 5)
            spo2 = np.random.normal(97, 1)
            temp = np.random.normal(36.8, 0.2)
            risk = np.random.uniform(0.0, 0.3)
        elif risk_level == "medium":
            hr = np.random.normal(85, 10)
            spo2 = np.random.normal(94, 2)
            temp = np.random.normal(37.2, 0.4)
            risk = np.random.uniform(0.3, 0.7)
        else:  # high
            hr = np.random.normal(100, 15)
            spo2 = np.random.normal(90, 3)
            temp = np.random.normal(37.8, 0.6)
            risk = np.random.uniform(0.7, 1.0)
        
        # Generate PPG signals (100 samples)
        red_signal = np.random.normal(1000, 100, 100).clip(500, 1500).astype(int).tolist()
        ir_signal = np.random.normal(800, 80, 100).clip(400, 1200).astype(int).tolist()
        
        return {
            'name': f'Patient {patient_id}',
            'red_signal': red_signal,
            'ir_signal': ir_signal,
            'temperature': float(temp),
            'heart_rate': float(hr),
            'spo2': float(spo2),
            'delirium_risk': float(risk)
        }
    
    def send_patient_update(self, patient_id: str, data: Dict) -> bool:
        """
        Send patient data to the webapp.
        
        Args:
            patient_id: Patient identifier
            data: Patient data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            endpoint = f"{self.webapp_url}/api/patients/{patient_id}/update"
            response = requests.post(endpoint, json=data, timeout=5)
            
            if response.status_code == 200:
                return True
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to webapp at {self.webapp_url}")
            return False
        except Exception as e:
            print(f"Error sending update: {e}")
            return False
    
    def populate_with_patients(self, num_patients: int = 5, num_updates: int = 1):
        """
        Populate the webapp with synthetic patients.
        
        Args:
            num_patients: Number of patients to create
            num_updates: Number of updates per patient
        """
        # Create patient distribution
        num_low = num_patients // 3
        num_medium = num_patients // 3
        num_high = num_patients - num_low - num_medium
        
        patients = []
        
        # Low risk patients
        for i in range(num_low):
            patients.append(('low_risk_' + str(i+1), 'low'))
        
        # Medium risk patients
        for i in range(num_medium):
            patients.append(('medium_risk_' + str(i+1), 'medium'))
        
        # High risk patients
        for i in range(num_high):
            patients.append(('high_risk_' + str(i+1), 'high'))
        
        print(f"\n{'='*60}")
        print("Webapp Patient Population Test")
        print(f"{'='*60}")
        print(f"\nTarget: {self.webapp_url}")
        print(f"Patients: {num_patients} ({num_low} low, {num_medium} medium, {num_high} high)")
        print(f"Updates per patient: {num_updates}")
        
        # Send updates
        total_updates = 0
        successful_updates = 0
        
        for update_num in range(num_updates):
            print(f"\n--- Update {update_num + 1}/{num_updates} ---")
            
            for patient_id, risk_level in patients:
                # Generate data
                data = self.generate_patient_data(patient_id, risk_level)
                
                # Send update
                success = self.send_patient_update(patient_id, data)
                total_updates += 1
                
                if success:
                    successful_updates += 1
                    status = "✓"
                else:
                    status = "✗"
                
                print(f"{status} {patient_id}: HR={data['heart_rate']:.0f} BPM, "
                      f"SpO2={data['spo2']:.1f}%, Temp={data['temperature']:.1f}°C, "
                      f"Risk={data['delirium_risk']*100:.1f}%")
            
            # Wait before next update (except for last one)
            if update_num < num_updates - 1:
                print(f"Waiting 5 seconds before next update...")
                time.sleep(5)
        
        # Summary
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        print(f"Total Updates: {total_updates}")
        print(f"Successful: {successful_updates}")
        print(f"Failed: {total_updates - successful_updates}")
        print(f"Success Rate: {100*successful_updates/total_updates:.1f}%")
        print(f"\nOpen browser to: {self.webapp_url}")
        print(f"{'='*60}\n")


def test_single_patient(webapp_url: str = "http://localhost:5000", patient_id: str = "test_patient"):
    """
    Test with a single patient.
    
    Args:
        webapp_url: Base URL of the web application
        patient_id: Patient identifier
    """
    print(f"\nTesting single patient: {patient_id}")
    print(f"Webapp URL: {webapp_url}\n")
    
    generator = SyntheticPatientGenerator(webapp_url)
    
    # Generate and send data
    data = generator.generate_patient_data(patient_id, 'medium')
    success = generator.send_patient_update(patient_id, data)
    
    if success:
        print(f"✓ Successfully sent patient data")
        print(f"  Heart Rate: {data['heart_rate']:.0f} BPM")
        print(f"  SpO2: {data['spo2']:.1f}%")
        print(f"  Temperature: {data['temperature']:.1f}°C")
        print(f"  Delirium Risk: {data['delirium_risk']*100:.1f}%")
    else:
        print(f"✗ Failed to send patient data")


def test_continuous_updates(webapp_url: str = "http://localhost:5000", 
                           patient_id: str = "continuous_patient",
                           duration_seconds: int = 60,
                           interval_seconds: int = 5):
    """
    Test with continuous patient updates.
    
    Args:
        webapp_url: Base URL of the web application
        patient_id: Patient identifier
        duration_seconds: How long to run the test
        interval_seconds: Interval between updates
    """
    print(f"\nTesting continuous updates")
    print(f"Patient: {patient_id}")
    print(f"Duration: {duration_seconds}s")
    print(f"Interval: {interval_seconds}s\n")
    
    generator = SyntheticPatientGenerator(webapp_url)
    start_time = time.time()
    update_count = 0
    
    while time.time() - start_time < duration_seconds:
        # Vary risk level over time
        elapsed = time.time() - start_time
        progress = elapsed / duration_seconds
        
        if progress < 0.33:
            risk_level = 'low'
        elif progress < 0.66:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        # Generate and send data
        data = generator.generate_patient_data(patient_id, risk_level)
        success = generator.send_patient_update(patient_id, data)
        
        update_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = "✓" if success else "✗"
        
        print(f"[{timestamp}] {status} Update {update_count}: "
              f"HR={data['heart_rate']:.0f}, SpO2={data['spo2']:.1f}%, "
              f"Risk={data['delirium_risk']*100:.1f}% ({risk_level})")
        
        # Wait for next update
        time.sleep(interval_seconds)
    
    print(f"\nTest complete. Total updates: {update_count}")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test webapp with synthetic patient data')
    parser.add_argument('--webapp', type=str, default='http://localhost:5000',
                       help='Webapp URL (default: http://localhost:5000)')
    parser.add_argument('--mode', type=str, default='populate',
                       choices=['populate', 'single', 'continuous'],
                       help='Test mode')
    parser.add_argument('--patients', type=int, default=5,
                       help='Number of patients (populate mode)')
    parser.add_argument('--updates', type=int, default=1,
                       help='Number of updates per patient (populate mode)')
    parser.add_argument('--patient-id', type=str, default='test_patient',
                       help='Patient ID (single/continuous mode)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in seconds (continuous mode)')
    parser.add_argument('--interval', type=int, default=5,
                       help='Update interval in seconds (continuous mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'populate':
        generator = SyntheticPatientGenerator(args.webapp)
        generator.populate_with_patients(args.patients, args.updates)
    elif args.mode == 'single':
        test_single_patient(args.webapp, args.patient_id)
    elif args.mode == 'continuous':
        test_continuous_updates(args.webapp, args.patient_id, args.duration, args.interval)


if __name__ == "__main__":
    main()
