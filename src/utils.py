"""
Utility functions for exoplanet detection and astronomical calculations
Includes visualization helpers, evaluation metrics, and domain-specific calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# Disposition class constants
DISPOSITION_CONFIRMED = 'CONFIRMED'
DISPOSITION_CANDIDATE = 'CANDIDATE'
DISPOSITION_FALSE_POSITIVE = 'FALSE_POSITIVE'
DISPOSITION_NO_PREDICT = 'NO_PREDICT'

# All valid disposition classes
VALID_DISPOSITIONS = [
    DISPOSITION_CONFIRMED,
    DISPOSITION_CANDIDATE,
    DISPOSITION_FALSE_POSITIVE,
    DISPOSITION_NO_PREDICT
]

# Training dispositions (exclude NO_PREDICT as it's for inference only)
TRAINING_DISPOSITIONS = [
    DISPOSITION_CONFIRMED,
    DISPOSITION_CANDIDATE,
    DISPOSITION_FALSE_POSITIVE
]


def validate_input_features(
    input_data: Dict[str, Any],
    required_features: Optional[List[str]] = None,
    min_features: int = 3
) -> Tuple[bool, str, str]:
    """
    Validate if user input has sufficient features for prediction
    
    Args:
        input_data: Dictionary of feature names and values
        required_features: List of feature names that must be present (optional)
        min_features: Minimum number of non-null features required
        
    Returns:
        Tuple of (is_valid, disposition, message)
        - is_valid: True if sufficient data, False otherwise
        - disposition: Predicted disposition or NO_PREDICT
        - message: Explanation message
    
    Example:
        >>> data = {'orbital_period': 10.5, 'planetary_radius': 2.3}
        >>> valid, disp, msg = validate_input_features(data, min_features=3)
        >>> if not valid:
        ...     print(f"Cannot predict: {msg}")
        ...     return disp  # Returns 'NO_PREDICT'
    """
    # Define core features needed for reliable prediction
    core_features = [
        'orbital_period',
        'transit_duration',
        'planetary_radius',
        'stellar_magnitude',
        'transit_depth'
    ]
    
    if required_features is None:
        required_features = core_features
    
    # Count non-null features
    non_null_count = 0
    missing_features = []
    
    for feature in required_features:
        value = input_data.get(feature)
        if value is not None and not pd.isna(value):
            non_null_count += 1
        else:
            missing_features.append(feature)
    
    # Check if we have minimum required features
    if non_null_count < min_features:
        message = (
            f"Insufficient data for prediction. "
            f"Provided {non_null_count}/{len(required_features)} required features. "
            f"Missing: {', '.join(missing_features)}. "
            f"Please provide at least {min_features} features."
        )
        return False, DISPOSITION_NO_PREDICT, message
    
    # Sufficient data available
    message = f"Valid input with {non_null_count}/{len(required_features)} features."
    return True, "", message


def check_feature_quality(
    input_data: Dict[str, Any],
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[bool, str]:
    """
    Check if feature values are within realistic ranges
    
    Args:
        input_data: Dictionary of feature names and values
        feature_ranges: Optional dict of (min, max) tuples for each feature
        
    Returns:
        Tuple of (is_valid, message)
        
    Example:
        >>> data = {'orbital_period': -5.0}  # Invalid: negative period
        >>> valid, msg = check_feature_quality(data)
        >>> if not valid:
        ...     print(msg)  # "Invalid orbital_period: must be > 0"
    """
    if feature_ranges is None:
        # Default realistic ranges for exoplanet features
        feature_ranges = {
            'orbital_period': (0.1, 100000),  # days
            'transit_duration': (0.01, 200),   # hours
            'planetary_radius': (0.1, 500),    # Earth radii
            'stellar_magnitude': (0, 25),      # magnitude
            'transit_depth': (0, 1000000),     # ppm
            'impact_parameter': (0, 2),        # dimensionless
            'equilibrium_temperature': (0, 10000),  # Kelvin
            'stellar_radius': (0.1, 500),      # Solar radii
            'stellar_mass': (0.01, 100)        # Solar masses
        }
    
    for feature, value in input_data.items():
        if value is None or pd.isna(value):
            continue
            
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            if not (min_val <= value <= max_val):
                return False, (
                    f"Invalid {feature}: {value}. "
                    f"Expected range: [{min_val}, {max_val}]"
                )
    
    return True, "All features within valid ranges"


def prepare_prediction_input(
    input_data: Dict[str, Any],
    required_features: List[str],
    fill_strategy: str = 'median',
    training_stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[np.ndarray, bool, str]:
    """
    Prepare user input for model prediction, handling missing values
    
    Args:
        input_data: Dictionary of feature names and values
        required_features: Ordered list of features expected by the model
        fill_strategy: Strategy for missing values ('median', 'mean', 'zero')
        training_stats: Statistics from training data (mean, median, std per feature)
        
    Returns:
        Tuple of (feature_array, is_valid, message)
        - feature_array: NumPy array ready for model.predict()
        - is_valid: True if prediction should proceed
        - message: Status or error message
    """
    # Validate input first
    is_valid, disposition, msg = validate_input_features(
        input_data,
        required_features=required_features
    )
    
    if not is_valid:
        return np.array([]), False, msg
    
    # Check feature quality
    quality_ok, quality_msg = check_feature_quality(input_data)
    if not quality_ok:
        return np.array([]), False, quality_msg
    
    # Build feature array
    features = []
    for feature in required_features:
        value = input_data.get(feature)
        
        if value is None or pd.isna(value):
            # Fill missing value based on strategy
            if training_stats and feature in training_stats:
                if fill_strategy == 'median':
                    value = training_stats[feature].get('median', 0)
                elif fill_strategy == 'mean':
                    value = training_stats[feature].get('mean', 0)
                else:
                    value = 0
            else:
                value = 0
        
        features.append(float(value))
    
    return np.array(features).reshape(1, -1), True, "Input prepared successfully"


def format_prediction_result(
    prediction: str,
    confidence: float,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format prediction results for web application response
    
    Args:
        prediction: Predicted disposition class
        confidence: Confidence score (0-1)
        input_data: Original input data
        
    Returns:
        Dictionary with formatted results
        
    Example:
        >>> result = format_prediction_result('CONFIRMED', 0.95, {...})
        >>> print(result['message'])
        "High confidence detection: Confirmed Exoplanet (95.0% confidence)"
    """
    # Handle NO_PREDICT case
    if prediction == DISPOSITION_NO_PREDICT:
        return {
            'disposition': DISPOSITION_NO_PREDICT,
            'confidence': 0.0,
            'message': 'Insufficient data for prediction. Please provide more features.',
            'recommendation': 'Add more observational data to improve prediction accuracy.',
            'status': 'no_prediction',
            'input_summary': input_data
        }
    
    # Format confidence levels
    if confidence >= 0.85:  # Changed from 0.9 to 0.85
        confidence_level = 'High'
        status = 'high_confidence'
    elif confidence >= 0.70:
        confidence_level = 'Moderate'
        status = 'moderate_confidence'
    else:
        confidence_level = 'Low'
        status = 'low_confidence'
    
    # Create human-readable labels
    disposition_labels = {
        DISPOSITION_CONFIRMED: 'Confirmed Exoplanet',
        DISPOSITION_CANDIDATE: 'Planet Candidate',
        DISPOSITION_FALSE_POSITIVE: 'False Positive'
    }
    
    disposition_descriptions = {
        DISPOSITION_CONFIRMED: 'This object is highly likely to be a real exoplanet.',
        DISPOSITION_CANDIDATE: 'This object shows planet-like characteristics but requires further validation.',
        DISPOSITION_FALSE_POSITIVE: 'This detection is likely a false positive (stellar activity, instrument noise, etc.).'
    }
    
    label = disposition_labels.get(prediction, prediction)
    description = disposition_descriptions.get(prediction, '')
    
    # Generate recommendation
    if prediction == DISPOSITION_CONFIRMED and confidence >= 0.9:
        recommendation = 'Recommend follow-up observations for detailed characterization.'
    elif prediction == DISPOSITION_CANDIDATE:
        recommendation = 'Recommend additional transit observations and radial velocity measurements.'
    elif prediction == DISPOSITION_FALSE_POSITIVE:
        recommendation = 'Suggest deprioritizing for follow-up. Consider checking for systematic errors.'
    else:
        recommendation = 'Collect more data to improve confidence level.'
    
    return {
        'disposition': prediction,
        'disposition_label': label,
        'confidence': float(confidence),
        'confidence_level': confidence_level,
        'message': f'{confidence_level} confidence detection: {label} ({confidence*100:.1f}% confidence)',
        'description': description,
        'recommendation': recommendation,
        'status': status,
        'input_summary': input_data
    }


# Example usage and documentation
if __name__ == "__main__":
    print("Exoplanet Detection Utilities")
    print("=" * 60)
    print("\nDisposition Classes:")
    for i, disp in enumerate(VALID_DISPOSITIONS, 1):
        print(f"  {i}. {disp}")
    
    print("\n" + "=" * 60)
    print("Example: Input Validation")
    print("=" * 60)
    
    # Example 1: Insufficient data
    insufficient_data = {
        'orbital_period': 10.5
    }
    valid, disp, msg = validate_input_features(insufficient_data)
    print(f"\nInput: {insufficient_data}")
    print(f"Valid: {valid}")
    print(f"Disposition: {disp}")
    print(f"Message: {msg}")
    
    # Example 2: Sufficient data
    sufficient_data = {
        'orbital_period': 10.5,
        'transit_duration': 3.2,
        'planetary_radius': 2.5,
        'stellar_magnitude': 12.3,
        'transit_depth': 500.0
    }
    valid, disp, msg = validate_input_features(sufficient_data)
    print(f"\nInput: {sufficient_data}")
    print(f"Valid: {valid}")
    print(f"Message: {msg}")
    
    # Example 3: Invalid feature values
    invalid_data = {
        'orbital_period': -5.0,  # Invalid: negative
        'planetary_radius': 2.5
    }
    valid, msg = check_feature_quality(invalid_data)
    print(f"\nInput: {invalid_data}")
    print(f"Valid: {valid}")
    print(f"Message: {msg}")
