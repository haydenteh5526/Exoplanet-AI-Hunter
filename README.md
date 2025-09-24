# Exoplanet AI Hunter 🚀🪐

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA-Space%20Apps%20Challenge%202025-blue)](https://spaceappschallenge.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Exoplanet%20Detection-green)](https://github.com)

An AI/ML system for automatically detecting exoplanets from NASA datasets (Kepler, TESS, K2). This project classifies astronomical observations as confirmed exoplanets, planetary candidates, or false positives using advanced machine learning techniques.

## 🎯 Project Overview

**Goal**: Create machine learning models that classify astronomical observations with >90% accuracy and <5% false positive rate, accessible through both expert and novice-friendly interfaces.

### Key Features
- 🤖 Multiple ML models (Random Forest, XGBoost, Neural Networks)
- 🌐 Web interface for data upload and real-time predictions
- 📊 Comprehensive data visualization and analysis
- 🔬 Scientifically accurate astronomical calculations
- 📈 Model interpretability and performance metrics
- 🚀 Production-ready deployment capabilities

## 🛠️ Technical Stack

- **Core**: Python 3.8+, pandas, numpy, scikit-learn
- **Deep Learning**: TensorFlow/Keras
- **ML Enhancement**: XGBoost, LightGBM, imbalanced-learn
- **Web Framework**: Flask with CORS support
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Astronomical**: Astropy for domain-specific calculations
- **Development**: Jupyter notebooks, pytest, black formatting

## 📊 Data Characteristics

- **Scale**: 10k-100k+ astronomical observations
- **Features**: orbital_period, transit_duration, planetary_radius, stellar_magnitude, transit_depth, impact_parameter, equilibrium_temperature, stellar_radius, stellar_mass
- **Classes**: CONFIRMED, CANDIDATE, FALSE_POSITIVE
- **Challenges**: Imbalanced classes, missing values, measurement uncertainties

## 🏗️ Project Structure

```
exoplanet-ai-hunter/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── data/                              # Data storage
│   ├── raw/                          # Original NASA datasets
│   ├── processed/                    # Cleaned datasets
│   └── sample/                       # Demo datasets
├── notebooks/                         # Jupyter analysis
│   ├── 01_data_exploration.ipynb     # Data exploration & EDA
│   ├── 02_feature_engineering.ipynb  # Feature creation & selection
│   └── 03_model_development.ipynb    # Model training & evaluation
├── src/                              # Core modules
│   ├── data_processing.py            # Data loading & preprocessing
│   ├── models.py                     # ML model implementations
│   └── utils.py                      # Utilities & astronomical functions
├── web_app/                          # Flask web application
│   ├── app.py                        # Main Flask application
│   ├── templates/                    # HTML templates
│   └── static/                       # CSS, JS, images
├── models/                           # Saved trained models
└── docs/                            # Documentation
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/exoplanet-ai-hunter.git
cd exoplanet-ai-hunter

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Download NASA datasets (instructions in docs/)
# Place raw datasets in data/raw/
# Run preprocessing pipeline
python src/data_processing.py
```

### 3. Model Training
```bash
# Explore data
jupyter notebook notebooks/01_data_exploration.ipynb

# Train models
python src/models.py --train --model all

# Evaluate performance
python src/models.py --evaluate
```

### 4. Web Application
```bash
# Start Flask development server
cd web_app
python app.py

# Open browser to http://localhost:5000
```

## 📈 Model Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Overall Accuracy | >90% | TBD |
| False Positive Rate | <5% | TBD |
| Precision (CONFIRMED) | >95% | TBD |
| Recall (CONFIRMED) | >85% | TBD |
| F1-Score | >90% | TBD |

## 🔬 Scientific Approach

### Astronomical Domain Knowledge
- **Transit Photometry**: Analysis of stellar brightness variations
- **Kepler's Laws**: Orbital mechanics validation
- **Stellar Classification**: Host star characterization
- **False Positive Mitigation**: Statistical significance testing

### Feature Engineering
- Derived astronomical parameters
- Temporal sequence analysis
- Multi-band photometry integration
- Uncertainty propagation

### Model Interpretability
- SHAP values for feature importance
- LIME for local explanations
- Astronomical parameter sensitivity analysis
- Decision boundary visualization

## 🌟 NASA Space Apps Challenge 2025

This project addresses the challenge of automated exoplanet detection using real NASA datasets. Our solution combines:

1. **Scientific Rigor**: Astronomically accurate methods and validations
2. **Technical Excellence**: Production-ready ML pipeline with >90% accuracy
3. **User Accessibility**: Intuitive interfaces for researchers and public
4. **Scalability**: Handles large-scale astronomical surveys (TESS, Kepler, K2)
5. **Innovation**: Novel feature engineering and ensemble methods

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NASA Exoplanet Archive for datasets
- Kepler, TESS, and K2 mission teams
- NASA Space Apps Challenge organizers
- Astronomical research community

## 📞 Contact

**Team**: NASA Space Apps Challenge 2025 Team
**Email**: [your-email@example.com]
**Project Link**: [https://github.com/yourusername/exoplanet-ai-hunter](https://github.com/yourusername/exoplanet-ai-hunter)

---

*"We are a way for the cosmos to know itself, and now we're teaching machines to help us discover new worlds."* 🌌
AI/ML tool for detecting exoplanets using NASA datasets - NASA Space Apps Challenge 2025
