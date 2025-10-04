# 📝 User Input Guide - Exoplanet AI Hunter

## How to Enter Data with Uncertainty Values

### ⚠️ Important Rule: Enter Only the Measured Value

When your astronomical data includes uncertainty (error margins), you should **only enter the measured value** (the number before the ± symbol).

---

## Examples

### Example 1: Transit Duration
**What you see in NASA data:**
```
Transit Duration [hrs]: 2.9575±0.0819
```

**What to enter in the form:**
```
2.9575
```

✅ **Correct:** `2.9575`  
❌ **Incorrect:** `2.9575±0.0819`  
❌ **Incorrect:** `2.9575 ± 0.0819`

---

### Example 2: Orbital Period
**What you see in NASA data:**
```
Orbital Period [days]: 1.736952453±2.63e-07
```

**What to enter in the form:**
```
1.736952453
```

✅ **Correct:** `1.736952453`  
❌ **Incorrect:** `1.736952453±2.63e-07`

---

### Example 3: Planetary Radius
**What you see in NASA data:**
```
Planetary Radius [Earth radii]: 2.26±0.15
```

**What to enter in the form:**
```
2.26
```

✅ **Correct:** `2.26`  
❌ **Incorrect:** `2.26±0.15`

---

## Understanding the ± Symbol

### What does ± mean?

The **±** symbol represents **uncertainty** or **margin of error**:

- **2.9575 ± 0.0819** means:
  - Measured value: **2.9575**
  - Uncertainty: **±0.0819**
  - True value likely between: **2.8756** and **3.0394**

### Why don't we enter the uncertainty?

The machine learning model is trained on **measured values only**. The uncertainty tells you about measurement quality but isn't used directly in prediction.

**Good to know:**
- ✅ Small uncertainty = High-quality data
- ⚠️ Large uncertainty = Lower-quality data

---

## Quick Reference Table

| Data Format | Enter This | Notes |
|-------------|------------|-------|
| `9.488±0.00003` | `9.488` | High precision ✅ |
| `2.96±0.08` | `2.96` | Good precision ✅ |
| `365.25±5.0` | `365.25` | Lower precision ⚠️ |
| `1500.5±100` | `1500.5` | Lower precision ⚠️ |
| `1.012345±0.000001` | `1.012345` | Ultra-high precision ✅ |

---

## Form Input Requirements

### Minimum Requirements
- Enter **at least 3 features** for prediction
- More features = better accuracy
- All values must be **numbers only**

### Supported Features
1. Orbital Period (days)
2. Transit Duration (hours)
3. Planetary Radius (Earth radii)
4. Stellar Magnitude
5. Transit Depth (ppm)
6. Impact Parameter
7. Equilibrium Temperature (K)
8. Insolation Flux (Earth flux)
9. Stellar Surface Gravity (log g)
10. Stellar Radius (Solar radii)
11. Stellar Mass (Solar masses)
12. Stellar Metallicity [Fe/H]

---

## Tips for Best Results

### ✅ DO:
- Enter only numeric values
- Use the measured value before ±
- Provide as many features as you have
- Use high-precision values when available
- Leave fields blank if you don't have the data

### ❌ DON'T:
- Include the ± symbol
- Include uncertainty values
- Enter text or units
- Make up values for missing data

---

## Example: Complete Input

**NASA Data for Kepler-227 b:**
```
Orbital Period: 9.488035±0.000028 days
Transit Duration: 2.96±0.08 hours
Planetary Radius: 2.26±0.15 Earth radii
Stellar Magnitude: 15.313
Transit Depth: 2089.0 ppm
Impact Parameter: 0.21
Equilibrium Temperature: 801.0 K
Stellar Radius: 0.88±0.03 Solar radii
Stellar Mass: 0.87±0.04 Solar masses
```

**What to enter in the form:**
```
Orbital Period: 9.488035
Transit Duration: 2.96
Planetary Radius: 2.26
Stellar Magnitude: 15.313
Transit Depth: 2089.0
Impact Parameter: 0.21
Equilibrium Temperature: 801.0
Stellar Radius: 0.88
Stellar Mass: 0.87
```

---

## Still Have Questions?

- The web form shows a blue information box with this reminder
- Hover over the ⓘ icon next to each field for specific help
- See `INPUT_DATA_GUIDE.md` for more detailed information

🌟 **Remember:** Always enter just the number before the ± symbol!
