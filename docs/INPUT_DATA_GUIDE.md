# 📊 Input Data Guide - Exoplanet AI Hunter

## Understanding Scientific Notation and Precision

### NASA Exoplanet Data Format

NASA exoplanet data often comes in this format:
```
Value ± Uncertainty
```

**Example:**
```
Orbital Period: 9.48803557±2.775e-05 days
```

This means:
- **Main value:** `9.48803557` days
- **Uncertainty:** `±2.775e-05` days (± 0.00002775 days)

### How to Enter Data in the Web App

**✅ DO:** Enter the main value with full precision
```
9.48803557
```

**❌ DON'T:** Try to enter the uncertainty
```
9.48803557±2.775e-05  ❌ (Invalid format)
```

---

## Real Example: Kepler-227 b

### Official NASA Data:
```
Orbital Period:            9.48803557±2.775e-05 days
Transit Duration:          2.96 hours
Planetary Radius:          2.26 Earth radii
Stellar Magnitude:         15.313
Transit Depth:             2089.0 ppm
Impact Parameter:          0.21
Equilibrium Temperature:   801.0 K
Stellar Radius:            0.88 Solar radii
Stellar Mass:              0.87 Solar masses
```

### How to Enter in the Form:
```
Orbital Period (days):              9.48803557
Transit Duration (hours):           2.96
Planetary Radius (Earth radii):     2.26
Stellar Magnitude:                  15.313
Transit Depth (ppm):                2089.0
Impact Parameter:                   0.21
Equilibrium Temperature (K):        801.0
Stellar Radius (Solar radii):       0.88
Stellar Mass (Solar masses):        0.87
```

### Expected Result:
- **Classification:** CONFIRMED
- **Confidence:** ~52%
- **Match:** K00752.01 (Kepler-227 b's catalog ID)
- **Similarity:** 99.9%

---

## Understanding Precision Levels

### High Precision (7-9 decimal places)
Examples from Kepler mission:
```
1.736952453     (Orbital period in days)
0.123456789     (Impact parameter)
15.31345678     (Stellar magnitude)
```

### Medium Precision (3-5 decimal places)
Common for calculated values:
```
801.123      (Temperature in Kelvin)
2.26789      (Planetary radius in Earth radii)
1.01234      (Stellar mass in Solar masses)
```

### Low Precision (1-2 decimal places)
Often sufficient for some measurements:
```
2089.0       (Transit depth in ppm)
2.96         (Transit duration in hours)
0.21         (Impact parameter)
```

---

## Scientific Notation Conversion

If you see scientific notation like `1.23e-05`, convert it to decimal:

| Scientific Notation | Decimal Form | Description |
|---------------------|--------------|-------------|
| `1.23e-05` | `0.0000123` | Very small value |
| `2.775e-05` | `0.00002775` | Small uncertainty |
| `1.5e+03` | `1500` | Large value |
| `9.48e+00` | `9.48` | Standard value |

---

## Minimum Data Requirements

The model requires **at least 3 features** for prediction, but more features improve accuracy:

### Minimum Input (3 features):
```
Orbital Period:     9.48803557
Planetary Radius:   2.26
Stellar Mass:       0.87
```
**Result:** Will work, but lower confidence

### Recommended Input (5-7 features):
```
Orbital Period:     9.48803557
Transit Duration:   2.96
Planetary Radius:   2.26
Equilibrium Temp:   801.0
Stellar Radius:     0.88
Stellar Mass:       0.87
Stellar Magnitude:  15.313
```
**Result:** Better confidence and more reliable

### Full Input (All 9 features):
**Best results and highest confidence!**

---

## Common NASA Data Sources

### Where to Find Real Exoplanet Data:

1. **NASA Exoplanet Archive**
   - URL: https://exoplanetarchive.ipac.caltech.edu/
   - Most comprehensive source
   - Download CSV files with all parameters

2. **Kepler Mission Data**
   - URL: https://archive.stsci.edu/kepler/
   - High-precision transit data
   - Includes KOI (Kepler Object of Interest) identifiers

3. **TESS Mission Data**
   - URL: https://tess.mit.edu/
   - TOI (TESS Object of Interest) identifiers
   - Recent discoveries

---

## Feature Descriptions

### Orbital Characteristics:
- **Orbital Period:** Time for one complete orbit (days)
- **Transit Duration:** Time planet blocks star's light (hours)
- **Impact Parameter:** How centrally planet crosses star (0-1)

### Planet Properties:
- **Planetary Radius:** Size relative to Earth (Earth radii)
- **Equilibrium Temperature:** Expected surface temperature (Kelvin)

### Transit Signal:
- **Transit Depth:** Brightness decrease during transit (ppm = parts per million)
- **Stellar Magnitude:** Star's apparent brightness

### Host Star Properties:
- **Stellar Radius:** Star size relative to Sun (Solar radii)
- **Stellar Mass:** Star mass relative to Sun (Solar masses)
- **Stellar Metallicity:** Metal content vs Sun ([Fe/H])
- **Stellar Surface Gravity:** Surface gravity (log g)

---

## Tips for Best Results

### ✅ DO:
- Enter as many features as you have
- Use full precision from NASA data
- Copy-paste values to avoid typos
- Leave unknown fields empty

### ❌ DON'T:
- Round values unnecessarily
- Make up values for missing data
- Enter text in number fields
- Include units (they're already labeled)

---

## Example Test Cases

### Test Case 1: Known Confirmed Exoplanet
```json
{
  "orbital_period": 9.48803557,
  "planetary_radius": 2.26,
  "stellar_mass": 0.87
}
```
Expected: CONFIRMED

### Test Case 2: Candidate with Uncertainty
```json
{
  "orbital_period": 15.5,
  "planetary_radius": 1.5,
  "equilibrium_temperature": 500.0
}
```
Expected: CANDIDATE

### Test Case 3: Likely False Positive
```json
{
  "orbital_period": 0.5,
  "planetary_radius": 0.1,
  "transit_depth": 10.0
}
```
Expected: FALSE_POSITIVE (unusual values)

---

## Troubleshooting

### "NO_PREDICT" Result
**Cause:** Not enough features provided (< 3)
**Solution:** Add more feature values

### Low Confidence (<40%)
**Cause:** Borderline case or missing key features
**Solution:** Add more features, especially:
- Orbital period
- Planetary radius
- Transit depth

### No Match Found
**Cause:** Input doesn't closely match any known exoplanet
**Solution:** This is normal! Not all candidates are in the database

---

## Additional Resources

- **NASA Exoplanet Archive:** https://exoplanetarchive.ipac.caltech.edu/
- **Exoplanet Encyclopedia:** http://exoplanet.eu/
- **Open Exoplanet Catalogue:** http://www.openexoplanetcatalogue.com/
- **TESS Mission:** https://tess.mit.edu/

---

**Happy Exoplanet Hunting! 🔭🪐**
