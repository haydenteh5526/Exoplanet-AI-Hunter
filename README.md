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
