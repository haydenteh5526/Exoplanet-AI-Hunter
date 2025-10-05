"""
Example: Using NO_PREDICT disposition in web application
Demonstrates how to handle insufficient user input
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    validate_input_features,
    check_feature_quality,
    format_prediction_result,
    DISPOSITION_NO_PREDICT,
    DISPOSITION_CONFIRMED,
    DISPOSITION_CANDIDATE,
    DISPOSITION_FALSE_POSITIVE
)


def predict_exoplanet(user_input: dict) -> dict:
    """
    Simulate exoplanet prediction with NO_PREDICT handling
    
    Args:
        user_input: Dictionary of user-provided features
        
    Returns:
        Prediction result dictionary
    """
    print("\n" + "="*70)
    print("EXOPLANET PREDICTION SYSTEM")
    print("="*70)
    print(f"\nUser Input: {user_input}")
    
    # Step 1: Validate input
    is_valid, disposition, message = validate_input_features(
        user_input,
        min_features=3
    )
    
    if not is_valid:
        print(f"\n❌ {message}")
        result = format_prediction_result(
            prediction=DISPOSITION_NO_PREDICT,
            confidence=0.0,
            input_data=user_input
        )
        return result
    
    # Step 2: Check feature quality
    quality_ok, quality_msg = check_feature_quality(user_input)
    if not quality_ok:
        print(f"\n❌ {quality_msg}")
        result = format_prediction_result(
            prediction=DISPOSITION_NO_PREDICT,
            confidence=0.0,
            input_data=user_input
        )
        return result
    
    print(f"\n✅ {message}")
    print(f"✅ {quality_msg}")
    
    # Step 3: Make prediction (simulated)
    # In real application, this would be: prediction = model.predict(features)
    # For demo, we'll simulate based on planetary radius
    planetary_radius = user_input.get('planetary_radius', 0)
    
    if planetary_radius < 1.5:
        prediction = DISPOSITION_CONFIRMED
        confidence = 0.92
    elif planetary_radius < 4.0:
        prediction = DISPOSITION_CANDIDATE
        confidence = 0.78
    else:
        prediction = DISPOSITION_FALSE_POSITIVE
        confidence = 0.85
    
    # Step 4: Format result
    result = format_prediction_result(
        prediction=prediction,
        confidence=confidence,
        input_data=user_input
    )
    
    return result


# Example scenarios
if __name__ == "__main__":
    
    # Scenario 1: Insufficient data (only 1 feature)
    print("\n" + "🔴 SCENARIO 1: INSUFFICIENT DATA")
    print("-" * 70)
    insufficient_input = {
        'orbital_period': 10.5
    }
    result1 = predict_exoplanet(insufficient_input)
    print(f"\n📊 Result:")
    print(f"   Disposition: {result1['disposition']}")
    print(f"   Confidence: {result1['confidence']:.1%}")
    print(f"   Message: {result1['message']}")
    print(f"   Recommendation: {result1['recommendation']}")
    
    
    # Scenario 2: Invalid data (negative value)
    print("\n\n" + "🔴 SCENARIO 2: INVALID DATA")
    print("-" * 70)
    invalid_input = {
        'orbital_period': -5.0,  # Invalid: negative period
        'transit_duration': 3.2,
        'planetary_radius': 2.5,
        'stellar_magnitude': 12.3
    }
    result2 = predict_exoplanet(invalid_input)
    print(f"\n📊 Result:")
    print(f"   Disposition: {result2['disposition']}")
    print(f"   Message: {result2['message']}")
    
    
    # Scenario 3: Valid data - Small planet (CONFIRMED)
    print("\n\n" + "🟢 SCENARIO 3: VALID DATA - SMALL PLANET")
    print("-" * 70)
    valid_input_confirmed = {
        'orbital_period': 10.5,
        'transit_duration': 3.2,
        'planetary_radius': 1.2,  # Small Earth-like
        'stellar_magnitude': 12.3,
        'transit_depth': 500.0
    }
    result3 = predict_exoplanet(valid_input_confirmed)
    print(f"\n📊 Result:")
    print(f"   Disposition: {result3['disposition']}")
    print(f"   Label: {result3['disposition_label']}")
    print(f"   Confidence: {result3['confidence']:.1%}")
    print(f"   Confidence Level: {result3['confidence_level']}")
    print(f"   Message: {result3['message']}")
    print(f"   Description: {result3['description']}")
    print(f"   Recommendation: {result3['recommendation']}")
    
    
    # Scenario 4: Valid data - Medium planet (CANDIDATE)
    print("\n\n" + "🟡 SCENARIO 4: VALID DATA - MEDIUM PLANET")
    print("-" * 70)
    valid_input_candidate = {
        'orbital_period': 25.3,
        'transit_duration': 4.5,
        'planetary_radius': 2.8,  # Neptune-sized
        'stellar_magnitude': 13.1,
        'transit_depth': 1200.0
    }
    result4 = predict_exoplanet(valid_input_candidate)
    print(f"\n📊 Result:")
    print(f"   Disposition: {result4['disposition']}")
    print(f"   Label: {result4['disposition_label']}")
    print(f"   Confidence: {result4['confidence']:.1%}")
    print(f"   Message: {result4['message']}")
    print(f"   Recommendation: {result4['recommendation']}")
    
    
    # Scenario 5: Valid data - Large "planet" (FALSE_POSITIVE)
    print("\n\n" + "🔵 SCENARIO 5: VALID DATA - LARGE OBJECT")
    print("-" * 70)
    valid_input_false = {
        'orbital_period': 100.0,
        'transit_duration': 8.0,
        'planetary_radius': 15.0,  # Too large (likely stellar)
        'stellar_magnitude': 14.5,
        'transit_depth': 5000.0
    }
    result5 = predict_exoplanet(valid_input_false)
    print(f"\n📊 Result:")
    print(f"   Disposition: {result5['disposition']}")
    print(f"   Label: {result5['disposition_label']}")
    print(f"   Confidence: {result5['confidence']:.1%}")
    print(f"   Message: {result5['message']}")
    print(f"   Recommendation: {result5['recommendation']}")
    
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY: DISPOSITION CLASSES")
    print("="*70)
    print("\nThe system now supports 4 disposition classes:")
    print(f"  1. {DISPOSITION_CONFIRMED} - Real exoplanet detected")
    print(f"  2. {DISPOSITION_CANDIDATE} - Potential planet requiring validation")
    print(f"  3. {DISPOSITION_FALSE_POSITIVE} - Likely not a planet")
    print(f"  4. {DISPOSITION_NO_PREDICT} - Insufficient/invalid data")
    print("\nNO_PREDICT is returned when:")
    print("  • User provides fewer than 3 features")
    print("  • Feature values are outside valid ranges")
    print("  • Data quality is insufficient for reliable prediction")
    print("\nThis helps users understand when they need to provide more data!")
    print("="*70 + "\n")
