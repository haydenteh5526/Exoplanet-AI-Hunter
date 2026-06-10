# Exoplanet AI Hunter 🚀🪐

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA-Space%20Apps%20Challenge%202025-blue)](https://spaceappschallenge.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-73.7%25-brightgreen)](https://github.com)

An AI-powered web application for classifying exoplanet observations from NASA's Kepler, K2, and TESS missions. Using a Random Forest model trained on 9,487 samples, this tool classifies observations as confirmed exoplanets, planetary candidates, or false positives.

## Quick Start

```bash
# Clone and install
git clone https://github.com/haydenteh5526/Exoplanet-AI-Hunter.git
cd Exoplanet-AI-Hunter
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# Run
cd web_app
python app.py
# Open http://localhost:5000
```

## Docker

```bash
docker compose up --build
# Open http://localhost:5000
```

## Tests

```bash
pytest tests/ -v
```

## Input Parameters

The model accepts 9 astronomical observation parameters:

| Parameter | Unit | Description |
|-----------|------|-------------|
| Orbital Period | days | Time for one complete orbit |
| Transit Duration | hours | Time to cross the star's disk |
| Planetary Radius | Earth radii | Size relative to Earth |
| Transit Depth | ppm | Decrease in star brightness during transit |
| Impact Parameter | 0–1 | Distance between planet and star center |
| Equilibrium Temperature | K | Expected temperature from stellar radiation |
| Insolation Flux | Earth flux | Amount of stellar energy received |
| Stellar Surface Gravity | log g | Surface gravity of the host star |
| Stellar Radius | Solar radii | Size of host star relative to the Sun |

## Classification Output

- **CONFIRMED** 🟢 — Validated exoplanet with high confidence
- **CANDIDATE** 🟡 — Potential exoplanet requiring further observation
- **FALSE_POSITIVE** 🔴 — Not an exoplanet (stellar activity, binary stars, etc.)

## Project Structure

```
Exoplanet-AI-Hunter/
├── web_app/                    # Flask web application
│   ├── app.py                  # Main app + prediction API
│   ├── templates/index.html    # Frontend UI
│   └── static/                 # CSS + JS
├── src/                        # Core Python modules
│   ├── data_processing.py      # NASA data standardization
│   ├── models.py               # ML model training pipeline
│   └── utils.py                # Validation & utility functions
├── models/                     # Trained model artifacts (.pkl)
├── data/
│   ├── raw/                    # Original NASA CSV files
│   └── processed/              # Standardized datasets
├── requirements.txt
└── README.md
```

## Technical Stack

- **Backend**: Flask, scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3 (glass morphism design), vanilla JS, Chart.js
- **ML Model**: Random Forest (73.7% accuracy, 3-class classification)
- **Data Sources**: NASA Exoplanet Archive (Kepler, K2, TESS)

## Model Training

To retrain the model (requires additional dependencies):

```bash
pip install tensorflow xgboost imbalanced-learn matplotlib seaborn
python src/models.py
```

## Data Processing

To re-standardize the raw NASA data:

```bash
python src/data_processing.py
```

## NASA Space Apps Challenge 2025

This project was developed for the NASA Space Apps Challenge 2025, combining real NASA observational data with machine learning to make exoplanet detection accessible to everyone.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- NASA Exoplanet Archive for comprehensive datasets
- Kepler, K2, and TESS mission teams
- NASA Space Apps Challenge for inspiring innovation in space exploration
