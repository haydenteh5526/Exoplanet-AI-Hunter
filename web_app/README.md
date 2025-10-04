# 🌟 Exoplanet AI Hunter - Web Application

## Quick Start Guide

Your Exoplanet AI Hunter web application is now **LIVE** and ready to use!

## 🚀 Running the Application

### Basic Usage
```cmd
python web_app\app.py
```

The Flask server will start at:
- **Local URL**: http://127.0.0.1:5000
- **Network URL**: http://0.0.0.0:5000

Simply open the local URL in your web browser to start using the application!

### ⚙️ Configuration (Optional)

The application can be configured using environment variables. Create a `.env` file in the `web_app` directory:

```env
# Flask Server Settings
FLASK_PORT=5000
FLASK_HOST=0.0.0.0
FLASK_DEBUG=True

# Exoplanet Matching Settings
MATCHING_TOP_N=3
MATCHING_SIMILARITY_CONFIRMED=0.80
MATCHING_SIMILARITY_DEFAULT=0.85
```

**Configuration Options**:
- `FLASK_PORT`: Port number for the web server (default: 5000)
- `FLASK_HOST`: Host address (default: 0.0.0.0 for all interfaces)
- `FLASK_DEBUG`: Enable debug mode (default: True)
- `MATCHING_TOP_N`: Number of similar exoplanets to show (default: 3)
- `MATCHING_MIN_FEATURES`: Minimum features required for matching (default: 3)
- `MATCHING_SIMILARITY_CONFIRMED`: Similarity threshold for confirmed predictions (default: 0.80)
- `MATCHING_SIMILARITY_DEFAULT`: Default similarity threshold (default: 0.85)

See `.env.example` for a template.

## 📊 How to Use

### 1. Enter Exoplanet Features
The web interface provides an input form with 9 astronomical features:

- **Orbital Period** (days) - Time for one complete orbit
- **Transit Duration** (hours) - Time to cross the star's disk  
- **Planetary Radius** (Earth radii) - Size relative to Earth
- **Equilibrium Temperature** (K) - Expected temperature
- **Insolation Flux** (Earth flux) - Stellar energy received
- **Stellar Surface Gravity** (log g) - Host star's gravity
- **Stellar Radius** (Solar radii) - Star size vs. Sun
- **Stellar Mass** (Solar masses) - Star mass vs. Sun
- **Stellar Metallicity** [Fe/H] - Metal content vs. Sun

**Minimum Requirement**: Enter at least 3 features for a prediction (more features = better accuracy!)

### 2. Submit for Classification
Click the **"🔍 Classify Exoplanet"** button to get your prediction.

### 3. View Results
The application will display:

✅ **Classification Result**:
- CONFIRMED PLANET (confirmed exoplanet)
- CANDIDATE (potential exoplanet requiring follow-up)
- FALSE POSITIVE (not a real planet)
- NO_PREDICT (insufficient data provided)

📈 **Confidence Score**: Percentage confidence in the prediction

📊 **Probability Chart**: Bar chart showing probabilities for all classes

📉 **Feature Importance**: Chart showing which features matter most to the model

💡 **Recommendation**: Guidance on next steps or data quality

## 🎯 Example Usage

### Confirmed Planet Example
```
Orbital Period: 3.52 days
Transit Duration: 2.5 hours
Planetary Radius: 1.2 Earth radii
Equilibrium Temperature: 1500 K
Insolation Flux: 100 Earth flux
Stellar Surface Gravity: 4.5
Stellar Radius: 1.0
Stellar Mass: 1.0
Stellar Metallicity: 0.0
```

### Candidate Example
```
Orbital Period: 87.5 days
Planetary Radius: 2.8 Earth radii
Equilibrium Temperature: 800 K
```

### False Positive Example
```
Transit Duration: 0.5 hours
Planetary Radius: 0.3 Earth radii
Equilibrium Temperature: 3500 K
```

## 🔧 Technical Details

### Model Information
- **Algorithm**: Random Forest Classifier
- **Training Data**: 9,487 Kepler observations
- **Accuracy**: 73.66%
- **Class Balancing**: SMOTE applied
- **Features**: 9 standardized astronomical measurements

### API Endpoints

#### GET /api/model-info
Returns information about the loaded model.

**Response**:
```json
{
  "model_type": "random_forest",
  "training_date": "2025-10-04",
  "accuracy": 0.7366,
  "features": [...],
  "classes": ["CANDIDATE", "CONFIRMED", "FALSE_POSITIVE"]
}
```

#### POST /api/predict
Make a prediction based on input features.

**Request Body**:
```json
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
```

**Response**:
```json
{
  "disposition": "CONFIRMED",
  "confidence": 0.85,
  "message": "High confidence prediction",
  "all_probabilities": {
    "CONFIRMED": 0.85,
    "CANDIDATE": 0.10,
    "FALSE_POSITIVE": 0.05
  },
  "recommendation": "This observation shows strong characteristics...",
  "quality_score": 1.0,
  "quality_message": "Excellent data quality",
  "features_provided": ["orbital_period", "transit_duration", ...]
}
```

#### GET /api/feature-importance
Get feature importance rankings from the model.

**Response**:
```json
{
  "feature_importance": [
    {"feature": "orbital_period", "importance": 0.25},
    {"feature": "planetary_radius", "importance": 0.20},
    ...
  ]
}
```

## 🛠️ Running the Application

### Start the Server
```bash
python web_app/app.py
```

Or with virtual environment:
```bash
C:\venv\nasa\Scripts\python.exe web_app/app.py
```

### Stop the Server
Press `Ctrl+C` in the terminal

## 📁 File Structure

```
web_app/
├── app.py                  # Flask backend application
├── templates/
│   └── index.html          # Main HTML interface
└── static/
    ├── css/
    │   └── style.css       # Custom styling
    └── js/
        └── app.js          # Frontend JavaScript
```

## 🎨 Features

✅ **User-Friendly Interface**: Clean, intuitive design with helpful tooltips

✅ **Real-Time Validation**: Instant feedback on input requirements

✅ **Interactive Visualizations**: Dynamic charts using Chart.js

✅ **NO_PREDICT Handling**: Smart detection of insufficient data

✅ **Responsive Design**: Works on desktop, tablet, and mobile

✅ **Error Handling**: Graceful error messages and recovery

## 🚦 Next Steps for Improvement

While the current model (73.66% accuracy) is functional, here are ways to improve:

1. **Train Better Models**:
   - Try XGBoost or Neural Network
   - Tune hyperparameters with GridSearchCV
   - Combine all 3 datasets (18,396 observations)

2. **Feature Engineering**:
   - Create derived features (e.g., planet-star ratio)
   - Remove low-importance features
   - Handle missing values better

3. **User Experience**:
   - Add example buttons (click to fill form)
   - Show historical predictions
   - Export results to PDF/CSV

4. **Deployment**:
   - Deploy to cloud (Heroku, Azure, AWS)
   - Add user authentication
   - Create REST API documentation

## 📞 Support

For issues or questions:
- Check the console/terminal for error messages
- Ensure all dependencies are installed (`pip install -r requirements.txt`)
- Verify the model files exist in `models/` directory

## 📄 License

NASA Space Apps Challenge 2025 - Exoplanet AI Hunter

---

**Enjoy hunting for exoplanets! 🌍🔭✨**
