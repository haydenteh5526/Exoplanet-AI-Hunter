"""
Exoplanet AI Hunter - Flask Web Application
NASA Space Apps Challenge 2025

This web application uses machine learning to classify astronomical observations
as confirmed exoplanets, planetary candidates, or false positives.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from pathlib import Path

# Add src directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / 'src'))

from utils import (
    validate_input_features, 
    check_feature_quality, 
    prepare_prediction_input,
    format_prediction_result,
    DISPOSITION_CONFIRMED,
    DISPOSITION_CANDIDATE,
    DISPOSITION_FALSE_POSITIVE,
    DISPOSITION_NO_PREDICT
)

app = Flask(__name__)

# Configuration constants
DEFAULT_PORT = int(os.getenv('FLASK_PORT', 5000))
DEFAULT_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
DEBUG_MODE = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# Exoplanet matching configuration
MATCHING_TOP_N = int(os.getenv('MATCHING_TOP_N', 3))
MATCHING_SIMILARITY_THRESHOLD_CONFIRMED = float(os.getenv('MATCHING_SIMILARITY_CONFIRMED', 0.80))
MATCHING_SIMILARITY_THRESHOLD_DEFAULT = float(os.getenv('MATCHING_SIMILARITY_DEFAULT', 0.85))
MATCHING_MIN_FEATURES = int(os.getenv('MATCHING_MIN_FEATURES', 3))

# Model paths
MODEL_DIR = project_root / 'models'
MODEL_FILE = MODEL_DIR / 'random_forest_20251004_145147.pkl'
SCALER_FILE = MODEL_DIR / 'random_forest_20251004_145147_scaler.pkl'
ENCODER_FILE = MODEL_DIR / 'random_forest_20251004_145147_encoder.pkl'
METADATA_FILE = MODEL_DIR / 'random_forest_20251004_145147_metadata.json'

# Data paths for exoplanet database
DATA_DIR = project_root / 'data' / 'processed'
KEPLER_DATA = DATA_DIR / 'kepler_standardized.csv'
K2_DATA = DATA_DIR / 'k2_standardized.csv'
TESS_DATA = DATA_DIR / 'tess_standardized.csv'

# Global variables for loaded model
model = None
scaler = None
encoder = None
metadata = None
exoplanet_db = None

def load_model():
    """Load the trained model and preprocessing objects"""
    global model, scaler, encoder, metadata, exoplanet_db
    
    try:
        print("Loading model...")
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded: {MODEL_FILE}")
        
        print("Loading scaler...")
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded: {SCALER_FILE}")
        
        print("Loading encoder...")
        with open(ENCODER_FILE, 'rb') as f:
            encoder = pickle.load(f)
        print(f"✅ Encoder loaded: {ENCODER_FILE}")
        
        print("Loading metadata...")
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded: {METADATA_FILE}")
        
        # Load exoplanet database
        print("Loading exoplanet database...")
        dfs = []
        if KEPLER_DATA.exists():
            df = pd.read_csv(KEPLER_DATA)
            df['source'] = 'Kepler'
            dfs.append(df)
            print(f"  ✅ Loaded {len(df)} Kepler exoplanets")
        
        if K2_DATA.exists():
            df = pd.read_csv(K2_DATA)
            df['source'] = 'K2'
            dfs.append(df)
            print(f"  ✅ Loaded {len(df)} K2 exoplanets")
        
        if TESS_DATA.exists():
            df = pd.read_csv(TESS_DATA)
            df['source'] = 'TESS'
            dfs.append(df)
            print(f"  ✅ Loaded {len(df)} TESS exoplanets")
        
        if dfs:
            exoplanet_db = pd.concat(dfs, ignore_index=True)
            print(f"✅ Total exoplanet database: {len(exoplanet_db)} entries")
        else:
            print("⚠️ No exoplanet database files found")
        
        print("\n🚀 Model successfully loaded and ready!")
        return True
    
    except FileNotFoundError as e:
        print(f"❌ Error: Model file not found - {e}")
        print("Please train a model first by running: python src/models.py")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


def find_matching_exoplanet(input_data, top_n=None, similarity_threshold=None):
    """
    Find known exoplanets that match the input data
    
    Args:
        input_data: Dictionary of input features
        top_n: Number of top matches to return (default from config)
        similarity_threshold: Minimum similarity score (0-1) to consider a match (default from config)
        
    Returns:
        List of matching exoplanets with similarity scores
    """
    if top_n is None:
        top_n = MATCHING_TOP_N
    if similarity_threshold is None:
        similarity_threshold = MATCHING_SIMILARITY_THRESHOLD_DEFAULT
    
    if exoplanet_db is None or exoplanet_db.empty:
        return []
    
    # Feature names to compare (only those available in input)
    comparable_features = [
        'orbital_period', 'transit_duration', 'planetary_radius',
        'equilibrium_temperature', 'insolation_flux',
        'stellar_surface_gravity', 'stellar_radius', 'stellar_mass'
    ]
    
    # Filter to features that user provided
    provided_features = [f for f in comparable_features if f in input_data and input_data[f] is not None]
    
    if len(provided_features) < MATCHING_MIN_FEATURES:
        return []  # Need at least MATCHING_MIN_FEATURES features to match
    
    matches = []
    
    for idx, row in exoplanet_db.iterrows():
        # Only consider CONFIRMED planets for matching
        if row.get('disposition') != 'CONFIRMED':
            continue
        
        # Calculate similarity for each feature
        similarities = []
        actually_matched_features = []  # Track features that were actually compared
        
        for feature in provided_features:
            db_value = row.get(feature)
            input_value = input_data[feature]
            
            if pd.isna(db_value) or db_value == 0:
                continue  # Skip features with missing database values
            
            # Calculate percentage difference
            diff = abs(db_value - input_value) / abs(db_value)
            similarity = max(0, 1 - diff)  # Convert difference to similarity
            similarities.append(similarity)
            actually_matched_features.append(feature)  # Only add if we could compare
        
        if not similarities:
            continue
        
        # Average similarity across all compared features
        avg_similarity = np.mean(similarities)
        
        if avg_similarity >= similarity_threshold:
            # Get identifier - prefer object_id over index
            identifier = row.get('object_id', f"Object-{idx}")
            
            # Helper function to convert NaN to None for JSON serialization
            def safe_value(val):
                if pd.isna(val):
                    return None
                return float(val) if isinstance(val, (int, float, np.integer, np.floating)) else val
            
            match_info = {
                'name': identifier,  # This will be KOI/EPIC/TOI identifier (e.g., "K00752.01")
                'similarity': float(avg_similarity),
                'source': row.get('dataset', row.get('source', 'Unknown')),  # Use 'dataset' column
                'discovery_year': safe_value(row.get('discovery_year')),
                'features': {
                    'orbital_period': safe_value(row.get('orbital_period')),
                    'transit_duration': safe_value(row.get('transit_duration')),
                    'planetary_radius': safe_value(row.get('planetary_radius')),
                    'equilibrium_temperature': safe_value(row.get('equilibrium_temperature')),
                    'insolation_flux': safe_value(row.get('insolation_flux')),
                    'stellar_surface_gravity': safe_value(row.get('stellar_surface_gravity')),
                    'stellar_radius': safe_value(row.get('stellar_radius')),
                    'stellar_mass': safe_value(row.get('stellar_mass')),
                    'stellar_metallicity': safe_value(row.get('stellar_metallicity')),
                },
                'matched_features': actually_matched_features,  # Only features that were actually compared
                'num_matched': len(actually_matched_features),
                'features_provided': len(provided_features),  # Total features user provided
                'features_skipped': len(provided_features) - len(actually_matched_features)  # Features with N/A in DB
            }
            matches.append(match_info)
    
    # Sort by similarity score (descending)
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    return matches[:top_n]


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': metadata.get('model_type', 'Unknown'),
        'training_date': metadata.get('timestamp', 'Unknown'),
        'accuracy': metadata.get('accuracy', 0),
        'features': metadata.get('feature_names', []),
        'classes': list(encoder.classes_) if encoder else []
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on user input
    
    Expected JSON format:
    {
        "orbital_period": 3.52,
        "transit_duration": 2.5,
        "planetary_radius": 1.2,
        "equilibrium_temperature": 1500,
        "insolation_flux": 100,
        "stellar_surface_gravity": 4.5,
        "stellar_radius": 1.0,
        "stellar_mass": 1.0,
        "stellar_metallicity": 0.0
    }
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'The prediction model is not available. Please contact support.'
        }), 500
    
    try:
        # Get input data from request
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({
                'error': 'No input data provided',
                'message': 'Please provide exoplanet features in JSON format.'
            }), 400
        
        # Validate input features (returns 3 values: is_valid, disposition, message)
        is_valid, validation_disposition, validation_message = validate_input_features(input_data)
        
        if not is_valid:
            # Return NO_PREDICT response
            return jsonify({
                'disposition': validation_disposition,
                'confidence': 0.0,
                'message': validation_message,
                'all_probabilities': {},
                'recommendation': 'Please provide at least 3 valid features for prediction.'
            })
        
        # Check feature quality
        quality_ok, quality_message = check_feature_quality(input_data)
        
        if not quality_ok:
            # Return error if data quality is bad
            return jsonify({
                'disposition': DISPOSITION_NO_PREDICT,
                'confidence': 0.0,
                'message': quality_message,
                'all_probabilities': {},
                'recommendation': 'Please check your input values.'
            })
        
        # Prepare input for model - build feature array from input_data
        features = metadata.get('feature_names', [])
        
        if not features:
            return jsonify({
                'error': 'Model metadata missing feature names',
                'message': 'The model configuration is incomplete. Please retrain the model.'
            }), 500
        
        feature_array = []
        
        for feature in features:
            value = input_data.get(feature, 0.0)  # Use 0.0 for missing features
            if value is None:
                value = 0.0
            feature_array.append(float(value))
        
        # Scale features
        feature_array_scaled = scaler.transform([feature_array])
        
        # Make prediction
        prediction = model.predict(feature_array_scaled)[0]
        probabilities = model.predict_proba(feature_array_scaled)[0]
        
        # Get class names and probabilities
        classes = encoder.classes_
        prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        
        # Get predicted class name
        predicted_class = encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))
        
        # Calculate quality score based on number of features provided
        total_features = len(features)
        provided_features = len([v for v in input_data.values() if v is not None])
        quality_score = provided_features / total_features if total_features > 0 else 0
        
        # Find matching known exoplanets (if CONFIRMED)
        matches = []
        if predicted_class == 'CONFIRMED':
            matches = find_matching_exoplanet(
                input_data, 
                top_n=MATCHING_TOP_N, 
                similarity_threshold=MATCHING_SIMILARITY_THRESHOLD_CONFIRMED
            )
        
        # Format response
        response = format_prediction_result(
            predicted_class,
            confidence,
            input_data
        )
        
        # Add additional information to response
        response['all_probabilities'] = prob_dict
        response['quality_score'] = quality_score
        response['quality_message'] = quality_message
        response['features_provided'] = list(input_data.keys())
        response['input_summary'] = input_data  # Add input summary for comparison table
        
        # Add matching exoplanets if found
        if matches:
            response['matched_exoplanets'] = matches
            response['best_match'] = matches[0]  # Top match
        
        print(f"✅ Prediction successful: {predicted_class} ({confidence:.2%})")
        if matches:
            print(f"   🎯 Best match: {matches[0]['name']} ({matches[0]['similarity']:.1%} similarity)")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/api/test-predict', methods=['POST'])
def test_predict():
    """Debug endpoint to see raw prediction response"""
    try:
        input_data = request.get_json()
        response = predict()
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    """Get feature importance from the model"""
    if model is None or not hasattr(model, 'feature_importances_'):
        return jsonify({'error': 'Feature importance not available'}), 500
    
    try:
        features = metadata.get('feature_names', [])
        importances = model.feature_importances_
        
        # Create list of (feature, importance) tuples and sort
        feature_imp = [
            {'feature': features[i], 'importance': float(importances[i])}
            for i in range(len(features))
        ]
        feature_imp.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'feature_importance': feature_imp
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌟 EXOPLANET AI HUNTER - WEB APPLICATION")
    print("="*60 + "\n")
    
    # Load model on startup
    if load_model():
        print("\n" + "="*60)
        print("🚀 Starting Flask server...")
        print("="*60 + "\n")
        app.run(debug=DEBUG_MODE, host=DEFAULT_HOST, port=DEFAULT_PORT)
    else:
        print("\n❌ Failed to load model. Please train a model first.")
        print("Run: python src/models.py")
        sys.exit(1)
