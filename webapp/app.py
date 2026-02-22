#!/usr/bin/env python3
"""
Delirium Monitoring Web Application

Simple Flask app to view patient data and delirium risk probabilities.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import joblib
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessor import SensorPreprocessor

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'delirium-detection-secret'

# Global state
model = None
preprocessor = SensorPreprocessor()

# In-memory patient data storage
patients_data = {}


def load_model(model_path: str):
    """Load the trained ensemble model."""
    global model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def calculate_risk_heuristic(data: dict) -> float:
    """
    Calculate delirium risk using simple heuristic when model is unavailable.
    Based on vital sign abnormalities.
    
    Args:
        data: Patient data dictionary
        
    Returns:
        Risk score between 0 and 1
    """
    risk = 0.0
    
    # Heart rate risk
    hr = data.get('heart_rate', 75)
    if hr < 50 or hr > 120:
        risk += 0.2
    elif hr < 60 or hr > 100:
        risk += 0.1
    
    # SpO2 risk
    spo2 = data.get('spo2', 97)
    if spo2 < 90:
        risk += 0.3
    elif spo2 < 95:
        risk += 0.2
    
    # Temperature risk
    temp = data.get('temperature', 36.8)
    if temp < 36 or temp > 39:
        risk += 0.3
    elif temp < 36.5 or temp > 38:
        risk += 0.1
    
    # Delirium risk from test_webapp
    if 'delirium_risk' in data:
        risk = data['delirium_risk']
    
    return min(risk, 1.0)  # Cap at 1.0


@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')


@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Get list of all patients with their latest data."""
    patients_list = []
    
    for patient_id, data in patients_data.items():
        patient_info = {
            'id': patient_id,
            'name': data.get('name', f'Patient {patient_id}'),
            'last_updated': data.get('last_updated'),
            'delirium_risk': data.get('delirium_risk', 0.0),
            'heart_rate': data.get('heart_rate', 0),
            'spo2': data.get('spo2', 95.0),
            'temperature': data.get('temperature', 0.0),
            'risk_level': get_risk_level(data.get('delirium_risk', 0.0))
        }
        patients_list.append(patient_info)
    
    # Sort by risk (highest first)
    patients_list.sort(key=lambda x: x['delirium_risk'], reverse=True)
    
    return jsonify(patients_list)


@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Get detailed data for a specific patient."""
    if patient_id not in patients_data:
        return jsonify({'error': 'Patient not found'}), 404
    
    data = patients_data[patient_id]
    
    return jsonify({
        'id': patient_id,
        'name': data.get('name', f'Patient {patient_id}'),
        'last_updated': data.get('last_updated'),
        'delirium_risk': data.get('delirium_risk', 0.0),
        'heart_rate': data.get('heart_rate', 0),
        'spo2': data.get('spo2', 95.0),
        'temperature': data.get('temperature', 0.0),
        'risk_level': get_risk_level(data.get('delirium_risk', 0.0)),
        'history': data.get('history', [])
    })


@app.route('/api/patients/<patient_id>/update', methods=['POST'])
def update_patient(patient_id):
    """Update patient data with new sensor readings."""
    try:
        data = request.json
        
        # Extract sensor data
        red_signal = np.array(data.get('red_signal', []))
        ir_signal = np.array(data.get('ir_signal', []))
        temperature = data.get('temperature', 0.0)
        
        # Process features and make prediction
        delirium_risk = 0.0
        
        if len(red_signal) > 0 and len(ir_signal) > 0:
            try:
                features = preprocessor.extract_features(
                    red_signal,
                    ir_signal,
                    np.array([temperature])
                )
                
                # Make prediction if model is available
                if model is not None:
                    # predict_proba returns (1, 2) array: [prob_no_delirium, prob_delirium]
                    proba = model.predict_proba(features)
                    delirium_risk = float(proba[0, 1])  # Get probability of delirium (class 1)
                else:
                    # Fallback: use simple heuristic if model not available
                    delirium_risk = calculate_risk_heuristic(data)
            except Exception as e:
                print(f"Error in feature extraction or prediction: {e}")
                # Fallback to heuristic
                delirium_risk = calculate_risk_heuristic(data)
        
        # Update patient data
        if patient_id not in patients_data:
            patients_data[patient_id] = {
                'name': data.get('name', f'Patient {patient_id}'),
                'history': []
            }
        
        timestamp = datetime.now().isoformat()
        
        # Calculate SpO2 from red and IR signals
        spo2 = preprocessor._calculate_spo2(red_signal, ir_signal) if len(red_signal) > 0 and len(ir_signal) > 0 else 95.0
        
        patients_data[patient_id].update({
            'last_updated': timestamp,
            'delirium_risk': delirium_risk,
            'heart_rate': data.get('heart_rate', 0),
            'spo2': spo2,
            'temperature': temperature
        })
        
        # Add to history
        history_entry = {
            'timestamp': timestamp,
            'delirium_risk': delirium_risk,
            'heart_rate': data.get('heart_rate', 0),
            'spo2': spo2,
            'temperature': temperature
        }
        
        patients_data[patient_id]['history'].append(history_entry)
        
        # Keep only last 100 entries
        if len(patients_data[patient_id]['history']) > 100:
            patients_data[patient_id]['history'] = patients_data[patient_id]['history'][-100:]
        
        return jsonify({
            'success': True,
            'delirium_risk': delirium_risk,
            'risk_level': get_risk_level(delirium_risk)
        })
        
    except Exception as e:
        print(f"Error updating patient: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/patients/<patient_id>/history', methods=['GET'])
def get_patient_history(patient_id):
    """Get patient's history data."""
    if patient_id not in patients_data:
        return jsonify({'error': 'Patient not found'}), 404
    
    history = patients_data[patient_id].get('history', [])
    
    return jsonify({
        'patient_id': patient_id,
        'history': history[-50:]  # Return last 50 entries
    })


@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get summary statistics."""
    if not patients_data:
        return jsonify({
            'total_patients': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0,
            'average_risk': 0.0
        })
    
    risks = [data.get('delirium_risk', 0.0) for data in patients_data.values()]
    
    high_risk = sum(1 for r in risks if r > 0.7)
    medium_risk = sum(1 for r in risks if 0.3 <= r <= 0.7)
    low_risk = sum(1 for r in risks if r < 0.3)
    
    return jsonify({
        'total_patients': len(patients_data),
        'high_risk_count': high_risk,
        'medium_risk_count': medium_risk,
        'low_risk_count': low_risk,
        'average_risk': float(np.mean(risks)) if risks else 0.0
    })


def get_risk_level(risk_score: float) -> str:
    """Get risk level from score."""
    if risk_score > 0.7:
        return 'HIGH'
    elif risk_score > 0.3:
        return 'MEDIUM'
    else:
        return 'LOW'


def create_app(model_path: str = 'models/saved_models/delirium_ensemble.joblib'):
    """Create and initialize the Flask app."""
    # Load the model
    if not load_model(model_path):
        print("Warning: Could not load model. App will run but predictions won't work.")
    
    return app


if __name__ == '__main__':
    # Create app
    app = create_app()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
