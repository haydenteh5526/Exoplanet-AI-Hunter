# NO_PREDICT Disposition Class - Implementation Guide

## Overview

The **NO_PREDICT** disposition class has been added to handle cases where user input is insufficient or invalid for making a reliable exoplanet prediction.

## Disposition Classes

Your system now supports **4 disposition classes**:

| Class | Type | Description | When to Use |
|-------|------|-------------|-------------|
| `CONFIRMED` | Training + Inference | Confirmed exoplanet | ML prediction with high confidence |
| `CANDIDATE` | Training + Inference | Planet candidate | ML prediction for potential planets |
| `FALSE_POSITIVE` | Training + Inference | False detection | ML prediction for non-planets |
| `NO_PREDICT` | **Inference only** | Insufficient data | User provides too few/invalid features |

## When NO_PREDICT is Returned

The system returns `NO_PREDICT` when:

1. **Too Few Features**: User provides fewer than 3 required features
2. **Invalid Values**: Feature values are outside realistic ranges (e.g., negative orbital period)
3. **Poor Data Quality**: Missing critical features needed for prediction

## Usage in Web Application

### Backend (Flask) Example

```python
from src.utils import (
    validate_input_features,
    check_feature_quality,
    format_prediction_result,
    DISPOSITION_NO_PREDICT
)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    user_input = request.json
    
    # Validate input
    is_valid, disposition, message = validate_input_features(
        user_input,
        min_features=3
    )
    
    if not is_valid:
        # Return NO_PREDICT response
        result = format_prediction_result(
            prediction=DISPOSITION_NO_PREDICT,
            confidence=0.0,
            input_data=user_input
        )
        return jsonify(result), 400
    
    # Check feature quality
    quality_ok, quality_msg = check_feature_quality(user_input)
    if not quality_ok:
        result = format_prediction_result(
            prediction=DISPOSITION_NO_PREDICT,
            confidence=0.0,
            input_data=user_input
        )
        return jsonify(result), 400
    
    # Make prediction with your trained model
    prediction = model.predict(features)
    confidence = model.predict_proba(features).max()
    
    # Format result
    result = format_prediction_result(
        prediction=prediction,
        confidence=confidence,
        input_data=user_input
    )
    
    return jsonify(result), 200
```

### Frontend (JavaScript) Example

```javascript
async function predictExoplanet() {
    const userInput = {
        orbital_period: parseFloat(document.getElementById('orbital_period').value),
        transit_duration: parseFloat(document.getElementById('transit_duration').value),
        planetary_radius: parseFloat(document.getElementById('planetary_radius').value),
        stellar_magnitude: parseFloat(document.getElementById('stellar_magnitude').value),
        transit_depth: parseFloat(document.getElementById('transit_depth').value)
    };
    
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(userInput)
    });
    
    const result = await response.json();
    
    // Handle NO_PREDICT case
    if (result.disposition === 'NO_PREDICT') {
        showError(result.message, result.recommendation);
        return;
    }
    
    // Display prediction
    displayResult(result);
}

function showError(message, recommendation) {
    document.getElementById('error-box').innerHTML = `
        <div class="alert alert-warning">
            <h4>⚠️ Cannot Make Prediction</h4>
            <p><strong>Issue:</strong> ${message}</p>
            <p><strong>Solution:</strong> ${recommendation}</p>
        </div>
    `;
}
```

## Response Format

### NO_PREDICT Response

```json
{
  "disposition": "NO_PREDICT",
  "confidence": 0.0,
  "message": "Insufficient data for prediction. Please provide more features.",
  "recommendation": "Add more observational data to improve prediction accuracy.",
  "status": "no_prediction",
  "input_summary": {
    "orbital_period": 10.5
  }
}
```

### Valid Prediction Response

```json
{
  "disposition": "CONFIRMED",
  "disposition_label": "Confirmed Exoplanet",
  "confidence": 0.92,
  "confidence_level": "High",
  "message": "High confidence detection: Confirmed Exoplanet (92.0% confidence)",
  "description": "This object is highly likely to be a real exoplanet.",
  "recommendation": "Recommend follow-up observations for detailed characterization.",
  "status": "high_confidence",
  "input_summary": {
    "orbital_period": 10.5,
    "transit_duration": 3.2,
    "planetary_radius": 1.2,
    "stellar_magnitude": 12.3,
    "transit_depth": 500.0
  }
}
```

## Minimum Feature Requirements

For reliable predictions, users must provide **at least 3** of these core features:

1. `orbital_period` (days)
2. `transit_duration` (hours)
3. `planetary_radius` (Earth radii)
4. `stellar_magnitude`
5. `transit_depth` (ppm)

**Recommended:** Provide all 5 core features for best accuracy.

## Feature Validation Ranges

| Feature | Min | Max | Unit |
|---------|-----|-----|------|
| orbital_period | 0.1 | 100,000 | days |
| transit_duration | 0.01 | 200 | hours |
| planetary_radius | 0.1 | 500 | Earth radii |
| stellar_magnitude | 0 | 25 | magnitude |
| transit_depth | 0 | 1,000,000 | ppm |
| impact_parameter | 0 | 2 | dimensionless |
| equilibrium_temperature | 0 | 10,000 | Kelvin |
| stellar_radius | 0.1 | 500 | Solar radii |
| stellar_mass | 0.01 | 100 | Solar masses |

## User Experience Guidelines

### Clear Error Messages

❌ **Bad**: "Error: Invalid input"

✅ **Good**: "Insufficient data for prediction. Provided 1/5 required features. Missing: transit_duration, planetary_radius, stellar_magnitude, transit_depth. Please provide at least 3 features."

### Helpful Recommendations

When returning `NO_PREDICT`, always include:
- **What's missing**: List specific missing features
- **How to fix**: Clear instructions on what data to add
- **Why it matters**: Explain why more data improves accuracy

### UI Feedback

```html
<!-- Example: Missing Data Alert -->
<div class="alert alert-warning">
    <h4>⚠️ More Data Needed</h4>
    <p>You've provided 2 out of 5 core features.</p>
    <p><strong>Missing:</strong></p>
    <ul>
        <li>Transit Duration</li>
        <li>Planetary Radius</li>
        <li>Transit Depth</li>
    </ul>
    <p>Add at least 1 more feature to enable prediction.</p>
</div>
```

## Testing

Run the example script to see NO_PREDICT in action:

```bash
python examples/no_predict_example.py
```

This demonstrates:
- ✅ Insufficient data handling
- ✅ Invalid value detection
- ✅ Successful predictions with good data
- ✅ Proper response formatting

## Integration Checklist

- [ ] Import validation functions from `src/utils.py`
- [ ] Validate user input before prediction
- [ ] Return `NO_PREDICT` for insufficient/invalid data
- [ ] Format responses using `format_prediction_result()`
- [ ] Display clear error messages to users
- [ ] Provide actionable recommendations
- [ ] Test edge cases (0 features, 1 feature, invalid values)
- [ ] Update API documentation with NO_PREDICT responses

## Benefits

✅ **Better UX**: Users know exactly what's needed
✅ **Prevents errors**: Avoids unreliable predictions
✅ **Guides users**: Clear instructions for data collection
✅ **Professional**: Handles edge cases gracefully
✅ **Maintains accuracy**: Only predicts when data is sufficient

---

**Related Files:**
- `src/utils.py` - Validation functions
- `examples/no_predict_example.py` - Working examples
- `data/processed/README.md` - Disposition documentation

**Last Updated:** October 4, 2025
