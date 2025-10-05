# Exoplanet AI Hunter 🚀🪐

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA-Space%20Apps%20Challenge%202025-blue)](https://spaceappschallenge.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green)](https://github.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-73.7%25-brightgreen)](https://github.com)

An AI-powered web application for classifying exoplanet observations from NASA's Kepler, K2, and TESS missions. Using machine learning trained on 9,487 samples, this tool helps distinguish between confirmed exoplanets, planetary candidates, and false positives with professional-grade accuracy.

## 🎯 Project Overview

**Mission**: Harness machine learning to classify astronomical observations and help discover new worlds beyond our solar system.

### ✨ Key Features
- 🤖 **Random Forest ML Model** - 73.7% accuracy on real NASA data
- 🌐 **Premium Web Interface** - Modern glass morphism design with smooth animations
- 📊 **Real-time Predictions** - Instant classification with probability distributions
- � **Feature Importance Visualization** - Understand which parameters matter most
- 🎨 **Responsive Design** - Works beautifully on desktop, tablet, and mobile
- 🚀 **Production Ready** - Flask backend with RESTful API

## 🛠️ Technical Stack

### Backend
- **Framework**: Flask (Python 3.8+)
- **ML Model**: Random Forest Classifier (scikit-learn)
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib

### Frontend
- **HTML5** with semantic structure
- **CSS3** with glass morphism effects, animations, and responsive grid
- **Vanilla JavaScript** for dynamic interactions
- **Chart.js** for data visualizations

### Data Sources
- NASA Exoplanet Archive (Kepler Mission)
- K2 Mission Candidate Planets
- TESS Objects of Interest (TOI)

## 📊 Model Performance

- **Training Samples**: 9,487 observations
- **Overall Accuracy**: 73.7%
- **Features**: 9 astronomical parameters
- **Classes**: CONFIRMED, CANDIDATE, FALSE_POSITIVE

## 🏗️ Project Structure

```
Exoplanet-AI-Hunter/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── PREMIUM_UI_v9.md                  # UI enhancement documentation
├── LAYOUT_RESTRUCTURE_v9.1.md        # Layout changes documentation
├── data/                              # NASA datasets
│   ├── raw/                          # Original data files
│   │   ├── cumulative_2025.09.18_13.24.09.csv      # Kepler
│   │   ├── k2pandc_2025.09.18_13.24.20.csv         # K2
│   │   └── TOI_2025.09.18_13.24.15.csv             # TESS
│   └── processed/                    # Standardized data
│       ├── kepler_standardized.csv
│       ├── k2_standardized.csv
│       └── tess_standardized.csv
├── src/                              # Core Python modules
│   ├── data_processing.py            # Data preprocessing
│   ├── models.py                     # ML model training
│   └── utils.py                      # Utility functions
├── web_app/                          # Flask web application
│   ├── app.py                        # Main Flask app
│   ├── templates/
│   │   └── index.html               # Main page (v9.1)
│   └── static/
│       ├── css/
│       │   └── style.css            # Premium UI styles (v9.1)
│       └── js/
│           └── app.js               # Frontend logic
├── models/                           # Trained ML models
│   ├── random_forest_20251004_145147.pkl
│   ├── random_forest_20251004_145147_metadata.json
│   ├── random_forest_20251004_145147_scaler.pkl
│   └── random_forest_20251004_145147_encoder.pkl
└── docs/                            # Additional documentation
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/haydenteh5526/Exoplanet-AI-Hunter.git
cd Exoplanet-AI-Hunter

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
# Navigate to web app directory
cd web_app

# Start Flask server
python app.py

# Open browser to http://localhost:5000
```

The application will start on `http://localhost:5000` by default.

## 📝 Input Parameters

The model accepts 9 astronomical observation parameters:

1. **Orbital Period** (days) - Time for one complete orbit around the star
2. **Transit Duration** (hours) - Time to cross the star's disk
3. **Planetary Radius** (Earth radii) - Size relative to Earth
4. **Transit Depth** (ppm) - Decrease in star brightness during transit
5. **Impact Parameter** - Distance between planet and star center (0-1)
6. **Equilibrium Temperature** (K) - Expected temperature based on stellar radiation
7. **Insolation Flux** (Earth flux) - Amount of stellar energy received
8. **Stellar Surface Gravity** (log g) - Surface gravity of the host star
9. **Stellar Radius** (Solar radii) - Size of host star relative to our Sun

## 🎨 UI Features (v9.1)

### Premium Design Elements
- **Glass Morphism** - Frosted glass cards with backdrop blur
- **Smooth Animations** - Fade-ins, slides, pulses, and shimmer effects
- **3-Column Grid Form** - Efficient horizontal layout for all 9 parameters
- **Full-Width Sections** - Modern spacious design
- **Responsive Layout** - Adapts to desktop (3 columns), tablet (2 columns), and mobile (1 column)
- **Interactive Charts** - Probability distribution and feature importance visualizations
- **Premium Gradients** - Purple-indigo-blue color scheme matching space theme
- **Micro-interactions** - Hover effects, ripple buttons, pulsing icons

### Layout Structure
1. **Hero Section** - "A World Away: Hunting for Exoplanets with AI"
2. **Features Section** - Showcase of capabilities
3. **Feature Importance Chart** - Shows most influential parameters
4. **Configure Parameters** - 3-column input form
5. **Classification Results** - AI predictions with probability charts

## 🔬 How It Works

1. **User Input**: Enter astronomical observation data (minimum 3 parameters)
2. **Data Processing**: Values are standardized using pre-fitted scaler
3. **ML Prediction**: Random Forest model classifies the observation
4. **Results Display**: Shows classification, confidence level, and probability distribution
5. **Feature Analysis**: Visualizes which parameters influenced the decision

## 🔬 How It Works

1. **User Input**: Enter astronomical observation data (minimum 3 parameters)
2. **Data Processing**: Values are standardized using pre-fitted scaler
3. **ML Prediction**: Random Forest model classifies the observation
4. **Results Display**: Shows classification, confidence level, and probability distribution
5. **Feature Analysis**: Visualizes which parameters influenced the decision

## 📈 Classification Categories

- **CONFIRMED** 🟢 - Validated exoplanet with high confidence
- **CANDIDATE** 🟡 - Potential exoplanet requiring further observation
- **FALSE_POSITIVE** 🔴 - Not an exoplanet (stellar activity, binary stars, etc.)

## � NASA Space Apps Challenge 2025

This project was developed for the NASA Space Apps Challenge 2025, addressing the challenge of automated exoplanet detection. Our solution combines:

1. **Real NASA Data** - Trained on actual Kepler, K2, and TESS observations
2. **Machine Learning** - Random Forest classifier with 73.7% accuracy
3. **User-Friendly Interface** - Premium web UI accessible to everyone
4. **Educational Value** - Helps users understand exoplanet detection science
5. **Production Quality** - Professional-grade design and implementation

## 🌟 Future Enhancements

- [ ] Increase model accuracy with ensemble methods
- [ ] Add more mission data (JWST, future missions)
- [ ] Implement deep learning models (CNN, LSTM)
- [ ] Add user authentication and saved predictions
- [ ] Export results to PDF/CSV
- [ ] Add dark/light mode toggle
- [ ] Integrate real-time NASA API data
- [ ] Add parallax effects and 3D visualizations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA Exoplanet Archive** - For providing comprehensive exoplanet datasets
- **Kepler, K2, TESS Mission Teams** - For groundbreaking exoplanet discoveries
- **NASA Space Apps Challenge** - For inspiring innovation in space exploration
- **scikit-learn Community** - For excellent ML tools and documentation
- **Chart.js** - For beautiful data visualizations

## 📞 Contact & Links

- **GitHub**: [haydenteh5526/Exoplanet-AI-Hunter](https://github.com/haydenteh5526/Exoplanet-AI-Hunter)
- **NASA Space Apps**: [2025 Challenge](https://spaceappschallenge.org/)
- **NASA Exoplanet Archive**: [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu/)

## 📚 Documentation

- `PREMIUM_UI_v9.md` - Detailed UI design documentation
- `LAYOUT_RESTRUCTURE_v9.1.md` - Layout architecture explanation
- `docs/INPUT_DATA_GUIDE.md` - Parameter input guidelines
- `data/processed/COLUMN_REFERENCE.md` - Dataset column descriptions

## 🔧 Development

### Running Tests
```bash
# Verify data columns
python verify_columns.py
```

### Data Processing
```bash
# Process raw NASA data
python src/data_processing.py
```

### Model Training
```bash
# Train new model (if needed)
python src/models.py
```

---

<div align="center">

**🌌 "We are a way for the cosmos to know itself, and now we're teaching machines to help us discover new worlds." 🌌**

Made with ❤️ for NASA Space Apps Challenge 2025

</div>
