# Standardized Dataset Column Reference

## Overview
All three NASA exoplanet datasets (Kepler, K2, TESS) have been standardized to have the **SAME column structure**. This ensures uniform data processing and machine learning model training.

## Standard Columns (13 total)

| Column Name | Data Type | Unit | Description | Kepler | K2 | TESS |
|-------------|-----------|------|-------------|--------|-----|------|
| `orbital_period` | float | days | Orbital period of the planet | ✅ 100% | ✅ 98.5% | ✅ 98.6% |
| `transit_duration` | float | hours | Duration of planetary transit | ✅ 100% | ❌ 0% | ✅ 100% |
| `planetary_radius` | float | Earth radii | Radius of the planet | ✅ 96.2% | ⚠️ 74.7% | ✅ 93.5% |
| `stellar_magnitude` | float | - | Apparent magnitude of host star | ✅ 100% | ❌ 0% | ✅ 100% |
| `transit_depth` | float | ppm | Depth of transit in parts per million | ✅ 96.2% | ❌ 0% | ✅ 100% |
| `impact_parameter` | float | - | Impact parameter (0-2) | ✅ 96.2% | ❌ 0% | ❌ 0% |
| `equilibrium_temperature` | float | Kelvin | Equilibrium temperature of planet | ✅ 96.2% | ⚠️ 13.9% | ✅ 96.3% |
| `stellar_radius` | float | Solar radii | Radius of host star | ✅ 96.2% | ✅ 96.4% | ✅ 93.5% |
| `stellar_mass` | float | Solar masses | Mass of host star | ✅ 96.2% | ⚠️ 49.0% | ❌ 0% |
| `disposition_score` | float | 0-1 | Confidence score for disposition | ✅ 84.4% | ❌ 0% | ❌ 0% |
| `object_id` | string | - | Unique identifier (KOI/EPIC/TOI) | ✅ 100% | ✅ 100% | ✅ 100% |
| `dataset` | string | - | Source dataset (KEPLER/K2/TESS) | ✅ 100% | ✅ 100% | ✅ 100% |
| `disposition` | string | - | Classification label | ✅ 100% | ✅ 100% | ✅ 100% |

**Legend:**
- ✅ High completeness (>90%)
- ⚠️ Partial completeness (10-90%)
- ❌ Not available (0%)
- Percentages show non-null data availability

## Disposition Classes

All datasets use the same 3 standardized disposition values for training:

| Value | Description | Kepler | K2 | TESS | Usage |
|-------|-------------|--------|-----|------|-------|
| `CONFIRMED` | Confirmed exoplanet | 28.9% | 32.3% | 9.5% | Training & Prediction |
| `CANDIDATE` | Planet candidate | 20.8% | 54.3% | 73.7% | Training & Prediction |
| `FALSE_POSITIVE` | False positive detection | 50.3% | 13.5% | 16.8% | Training & Prediction |
| `NO_PREDICT` | Insufficient data for prediction | N/A | N/A | N/A | **Inference only** |

**Note:** `NO_PREDICT` is used by the ML model during inference when user input lacks sufficient features for reliable prediction. It is NOT present in the training datasets.

## Dataset Statistics

### Kepler Cumulative Dataset
- **Total Records:** 9,487
- **File:** `kepler_standardized.csv`
- **Feature Completeness:** 96-100% for most features
- **Disposition Score:** Available (84.4%)

### K2 Planets & Candidates Dataset
- **Total Records:** 1,798
- **File:** `k2_standardized.csv`
- **Feature Completeness:** Limited (missing transit_duration, stellar_magnitude, transit_depth, impact_parameter, disposition_score)
- **Best for:** Confirmed planets and basic orbital parameters

### TESS TOI Dataset
- **Total Records:** 7,111
- **File:** `tess_standardized.csv`
- **Feature Completeness:** High for most features except impact_parameter, stellar_mass, disposition_score
- **Best for:** Recent candidates and transit properties

## Missing Data Strategy

**For Machine Learning:**
- ✅ **Recommended:** Use Kepler dataset for full-featured models (best completeness)
- ⚠️ **Alternative:** Combine datasets but handle missing values with:
  - Imputation (mean/median for numerical features)
  - Feature importance analysis (drop low-importance features with high missingness)
  - Dataset-specific models (train separate models for each dataset)

**Missing Value Indicators:**
- All missing values are represented as `NaN` (Not a Number)
- Can be detected with `pd.isna()` or `pd.notna()` in pandas

## Data Quality Notes

1. **All datasets have been cleaned:**
   - Removed UNKNOWN dispositions
   - Removed duplicate object IDs
   - Removed physically impossible values (negative radii, etc.)

2. **Units are standardized:**
   - All orbital periods in days
   - All transit durations in hours
   - All radii in Earth/Solar radii
   - All temperatures in Kelvin
   - All transit depths in ppm

3. **Object IDs are unique within each dataset:**
   - Kepler: KOI names (e.g., K00752.01)
   - K2: EPIC IDs and planet names
   - TESS: TOI numbers (e.g., 1000.01)

## Usage Example

```python
import pandas as pd

# Load all three datasets
kepler_df = pd.read_csv('data/processed/kepler_standardized.csv')
k2_df = pd.read_csv('data/processed/k2_standardized.csv')
tess_df = pd.read_csv('data/processed/tess_standardized.csv')

# All have the same columns!
print(kepler_df.columns.tolist())
print(k2_df.columns.tolist())
print(tess_df.columns.tolist())

# Filter by disposition
confirmed_planets = kepler_df[kepler_df['disposition'] == 'CONFIRMED']
candidates = kepler_df[kepler_df['disposition'] == 'CANDIDATE']
false_positives = kepler_df[kepler_df['disposition'] == 'FALSE_POSITIVE']

# Check data completeness
print(kepler_df.notna().sum() / len(kepler_df) * 100)
```

## Next Steps for ML

1. **Feature Selection:** Choose features based on completeness and importance
2. **Data Splitting:** Use stratified split to maintain class balance
3. **Handle Missing Values:** Impute or drop features/rows as needed
4. **Feature Engineering:** Create derived features (e.g., planet density, stellar flux)
5. **Model Training:** Train on standardized features with consistent target labels

---

**Generated:** October 4, 2025  
**Data Sources:** NASA Exoplanet Archive (Kepler, K2, TESS)  
**Total Records:** 18,396 observations
