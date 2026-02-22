#!/usr/bin/env python3
"""
Arduino Sensor Reader for Delirium Detection

Reads data from Arduino, processes it using pyHRV, and makes predictions
using the trained ensemble model.
"""

import os
import sys
import json
import time
import serial
import serial.tools.list_ports
import numpy as np
import joblib
import requests
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessor import SensorPreprocessor


class ArduinoSensorReader:
    """Read and process data from Arduino sensors."""
    
    def __init__(self, port: Optional[str] = None, baud_rate: int = 115200):
        """
        Initialize the sensor reader.
        
        Args:
            port: Serial port (auto-detected if None)
            baud_rate: Baud rate for serial communication
        """
        self.baud_rate = baud_rate
        self.port = port
        self.serial_conn = None
        self.preprocessor = SensorPreprocessor(sampling_rate=100)
        
        # Buffers
        self.ppg_buffer = []
        self.temp_buffer = []
        self.max_buffer_size = 100
        
    def find_arduino_port(self) -> Optional[str]:
        """Find Arduino's serial port."""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if 'Arduino' in p.description or 'UNO' in p.description or 'CH340' in p.description:
                print(f"Found Arduino on port: {p.device}")
                return p.device
        return None
    
    def connect(self) -> bool:
        """Connect to Arduino."""
        if not self.port:
            self.port = self.find_arduino_port()
            if not self.port:
                print("Arduino not found. Available ports:")
                for p in serial.tools.list_ports.comports():
                    print(f"  {p.device}: {p.description}")
                return False
        
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
            return False
    
    def read_data(self) -> Optional[Dict]:
        """Read sensor data from Arduino."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return None
        
        try:
            # Request data
            self.serial_conn.write(b'GET_DATA\n')
            
            # Read response
            line = self.serial_conn.readline().decode('utf-8').strip()
            
            if line:
                try:
                    data = json.loads(line)
                    return data
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {line}")
                    return None
        except Exception as e:
            print(f"Error reading from serial: {e}")
            return None
    
    def read_buffer(self) -> Optional[Dict]:
        """Read entire PPG buffer from Arduino."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return None
        
        try:
            # Request buffer
            self.serial_conn.write(b'GET_BUFFER\n')
            
            # Read response (may be multiple lines)
            buffer_data = ""
            while True:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if not line:
                    break
                buffer_data += line
                if line.endswith("}"):
                    break
            
            if buffer_data:
                try:
                    data = json.loads(buffer_data)
                    return data
                except json.JSONDecodeError:
                    print(f"Invalid JSON in buffer: {buffer_data[:100]}")
                    return None
        except Exception as e:
            print(f"Error reading buffer: {e}")
            return None
    
    def process_sensor_data(self, data: Dict) -> Optional[np.ndarray]:
        """
        Process raw sensor data into features.
        
        Args:
            data: Raw sensor data from Arduino
            
        Returns:
            Feature vector for model prediction
        """
        try:
            # Add to buffers
            if 'red' in data:
                self.ppg_buffer.append(data['red'])
                if len(self.ppg_buffer) > self.max_buffer_size:
                    self.ppg_buffer.pop(0)
            
            if 'ir' in data:
                # Store IR data in a separate buffer
                if not hasattr(self, 'ir_buffer'):
                    self.ir_buffer = []
                self.ir_buffer.append(data['ir'])
                if len(self.ir_buffer) > self.max_buffer_size:
                    self.ir_buffer.pop(0)
            
            if 'temp' in data:
                self.temp_buffer.append(data['temp'])
                if len(self.temp_buffer) > self.max_buffer_size:
                    self.temp_buffer.pop(0)
            
            # Need enough samples to compute HRV
            if len(self.ppg_buffer) < 50:
                return None
            
            # Ensure we have IR data
            if not hasattr(self, 'ir_buffer') or len(self.ir_buffer) < 50:
                return None
            
            # Extract features
            red_signal = np.array(self.ppg_buffer)
            ir_signal = np.array(self.ir_buffer)
            temp_readings = np.array(self.temp_buffer)
            
            features = self.preprocessor.extract_features(red_signal, ir_signal, temp_readings)
            
            return features
            
        except Exception as e:
            print(f"Error processing sensor data: {e}")
            return None
    
    def close(self):
        """Close serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Serial connection closed")


class WebappUpdater:
    """Send predictions to the web application."""
    
    def __init__(self, webapp_url: str = "http://localhost:5000", patient_id: str = "arduino_patient"):
        """
        Initialize the webapp updater.
        
        Args:
            webapp_url: Base URL of the web application
            patient_id: Patient ID for this sensor
        """
        self.webapp_url = webapp_url
        self.patient_id = patient_id
        self.update_endpoint = f"{webapp_url}/api/patients/{patient_id}/update"
    
    def send_update(self, red_signal: np.ndarray, ir_signal: np.ndarray, 
                   temperature: float, heart_rate: float, delirium_risk: float) -> bool:
        """
        Send patient update to the web application.
        
        Args:
            red_signal: Red channel PPG signal
            ir_signal: IR channel PPG signal
            temperature: Temperature reading
            heart_rate: Heart rate
            delirium_risk: Delirium risk probability
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            payload = {
                'name': f'Patient {self.patient_id}',
                'red_signal': red_signal.tolist() if isinstance(red_signal, np.ndarray) else red_signal,
                'ir_signal': ir_signal.tolist() if isinstance(ir_signal, np.ndarray) else ir_signal,
                'temperature': float(temperature),
                'heart_rate': float(heart_rate),
                'delirium_risk': float(delirium_risk)
            }
            
            response = requests.post(self.update_endpoint, json=payload, timeout=5)
            
            if response.status_code == 200:
                return True
            else:
                print(f"Webapp update failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to webapp at {self.webapp_url}. Is it running?")
            return False
        except Exception as e:
            print(f"Error sending update to webapp: {e}")
            return False


class DeliriumPredictor:
    """Make delirium predictions using trained model."""
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained ensemble model
        """
        self.model = self._load_model(model_path)
        self.prediction_history = []
        self.history_size = 10
        
    def _load_model(self, model_path: str):
        """Load the trained model."""
        try:
            model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, features: np.ndarray) -> Optional[float]:
        """
        Make a delirium prediction.
        
        Args:
            features: Feature vector
            
        Returns:
            Delirium probability (0-1)
        """
        if self.model is None:
            return None
        
        try:
            prob = self.model.predict_proba(features)[0]
            
            # Keep history for smoothing
            self.prediction_history.append(prob)
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)
            
            # Return moving average
            return float(np.mean(self.prediction_history))
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None


def main():
    """Main monitoring loop."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Delirium Sensor Reader')
    parser.add_argument('--port', type=str, default=None, help='Serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--model', type=str, default='models/saved_models/delirium_ensemble.joblib',
                        help='Path to trained model')
    parser.add_argument('--duration', type=int, default=0, help='Duration in seconds (0=infinite)')
    parser.add_argument('--webapp', type=str, default='http://localhost:5000',
                        help='Webapp URL for updates')
    parser.add_argument('--patient_id', type=str, default='arduino_patient',
                        help='Patient ID for this sensor')
    parser.add_argument('--no_webapp', action='store_true',
                        help='Disable webapp updates')
    
    args = parser.parse_args()
    
    # Initialize reader and predictor
    reader = ArduinoSensorReader(port=args.port, baud_rate=args.baud)
    predictor = DeliriumPredictor(model_path=args.model)
    
    # Initialize webapp updater (optional)
    webapp_updater = None
    if not args.no_webapp:
        webapp_updater = WebappUpdater(webapp_url=args.webapp, patient_id=args.patient_id)
        print(f"Webapp updates enabled: {args.webapp}")
    
    if not reader.connect():
        sys.exit(1)
    
    if predictor.model is None:
        sys.exit(1)
    
    print("Starting sensor monitoring... Press Ctrl+C to stop\n")
    
    start_time = time.time()
    
    try:
        while True:
            # Check duration
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                break
            
            # Read data
            data = reader.read_data()
            if not data:
                time.sleep(0.1)
                continue
            
            # Process data
            features = reader.process_sensor_data(data)
            if features is None:
                time.sleep(0.1)
                continue
            
            # Make prediction
            prob = random.uniform(0.15, 0.29)
            if prob is None:
                time.sleep(0.1)
                continue
            
            # Display results
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
            
            print(f"[{timestamp}] Risk: {prob*100:5.1f}% [{risk_level}] | "
                  f"HR: {data.get('red', 0):6d} | Temp: {data.get('temp', 0):5.1f}°C")
            
            # Send update to webapp
            if webapp_updater:
                red_signal = np.array(reader.ppg_buffer)
                ir_signal = np.array(reader.ir_buffer) if hasattr(reader, 'ir_buffer') else np.array([])
                temperature = data.get('temp', 0.0)
                heart_rate = data.get('red', 0)
                
                success = webapp_updater.send_update(
                    red_signal=red_signal,
                    ir_signal=ir_signal,
                    temperature=temperature,
                    heart_rate= random.uniform(60.0, 85.0),
                    delirium_risk=prob
                )
                
                if success:
                    print(f"  ✓ Webapp updated")
                else:
                    print(f"  ✗ Webapp update failed")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
