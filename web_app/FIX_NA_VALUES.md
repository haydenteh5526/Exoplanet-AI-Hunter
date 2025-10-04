# 🔧 Fix: Handling N/A Database Values in Similarity Matching

## Problem

When comparing user input to database exoplanets, some database entries have missing (N/A) values for certain features. This caused:

1. ❌ Comparison table showing "N/A" rows
2. ❌ Inaccurate "Features Matched" count
3. ❌ Confusion about which features were actually compared
4. ❌ Similarity calculation issues

## Example of the Problem

**User Input:**
```json
{
  "orbital_period": 9.49,
  "transit_duration": 2.96,
  "planetary_radius": 2.26,
  "equilibrium_temperature": 793.0,
  "insolation_flux": 93.59,
  "stellar_surface_gravity": 4.47,
  "stellar_radius": 0.93,
  "stellar_mass": 0.87,
  "stellar_metallicity": 0.1
}
```
**User provided:** 9 features

**Database (Kepler-227 b):**
```csv
orbital_period: 9.488
transit_duration: 2.9575
planetary_radius: 2.26
equilibrium_temperature: 793.0
insolation_flux: N/A        ← Missing
stellar_surface_gravity: N/A ← Missing
stellar_radius: 0.927
stellar_mass: 0.919
stellar_metallicity: N/A     ← Missing
```
**Database has:** 6 valid values, 3 N/A values

### Before Fix (Incorrect Display):

```
Features Matched: 9/9

Comparison Table:
Feature                    | Your Input | Database Value
Orbital Period            | 9.49       | 9.49
Transit Duration          | 2.96       | 2.96
Planetary Radius          | 2.26       | 2.26
Equilibrium Temperature   | 793.00     | 793.00
Insolation Flux          | 93.59      | N/A          ❌
Stellar Surface Gravity   | 4.47       | N/A          ❌
Stellar Radius            | 0.93       | 0.93
Stellar Mass              | 0.87       | 0.87
Stellar Metallicity       | 0.10       | N/A          ❌
```

**Issues:**
- Says "9/9 matched" but only 6 were actually compared
- Shows confusing N/A rows
- User wonders why it says "100% match" when 3 features are N/A

---

## Solution

### 1. ✅ Track Actually Compared Features

**Before:**
```python
similarities = []
for feature in provided_features:
    db_value = row.get(feature)
    if pd.isna(db_value) or db_value == 0:
        continue  # Skip but still counted in matched_features
    similarities.append(similarity)

match_info['matched_features'] = provided_features  # Wrong!
```

**After:**
```python
similarities = []
actually_matched_features = []  # NEW: Track what we compared

for feature in provided_features:
    db_value = row.get(feature)
    if pd.isna(db_value) or db_value == 0:
        continue  # Skip features with missing DB values
    
    similarities.append(similarity)
    actually_matched_features.append(feature)  # Only add if compared

match_info['matched_features'] = actually_matched_features  # Correct!
match_info['features_provided'] = len(provided_features)
match_info['features_skipped'] = len(provided_features) - len(actually_matched_features)
```

---

### 2. ✅ Update Display to Show Accurate Count

**Before:**
```javascript
Features Matched: 9/9
```

**After:**
```javascript
Features Compared: 6/9
Note: 3 feature(s) not available in database for this object
```

---

### 3. ✅ Only Show Compared Features in Table

The comparison table now only displays features that were actually compared (have values in both user input AND database).

**After Fix (Correct Display):**

```
Features Compared: 6/9
Note: 3 feature(s) not available in database for this object

Comparison Table:
Feature                    | Your Input | Database Value
Orbital Period            | 9.49       | 9.49
Transit Duration          | 2.96       | 2.96
Planetary Radius          | 2.26       | 2.26
Equilibrium Temperature   | 793.00     | 793.00
Stellar Radius            | 0.93       | 0.93
Stellar Mass              | 0.87       | 0.87
```

**Benefits:**
- ✅ Accurate count: "6/9" instead of "9/9"
- ✅ Clear explanation: "3 features not available"
- ✅ Clean table: No confusing N/A rows
- ✅ Honest similarity: Based on 6 features, not 9

---

## How Similarity is Calculated

### Formula

For each feature where **both** user input AND database have values:

```python
difference = abs(user_value - db_value) / abs(db_value)
similarity = 1 - difference
```

Average similarity across all compared features:

```python
avg_similarity = mean(all_feature_similarities)
```

### Example Calculation

**Compared Features (6 out of 9):**

| Feature | User | Database | Difference | Similarity |
|---------|------|----------|------------|------------|
| Orbital Period | 9.49 | 9.488 | 0.02% | 99.98% |
| Transit Duration | 2.96 | 2.9575 | 0.08% | 99.92% |
| Planetary Radius | 2.26 | 2.26 | 0% | 100% |
| Equilibrium Temp | 793.0 | 793.0 | 0% | 100% |
| Stellar Radius | 0.93 | 0.927 | 0.32% | 99.68% |
| Stellar Mass | 0.87 | 0.919 | 5.63% | 94.37% |

**Skipped Features (3 - N/A in database):**
- Insolation Flux
- Stellar Surface Gravity  
- Stellar Metallicity

**Final Similarity:**
```
(99.98 + 99.92 + 100 + 100 + 99.68 + 94.37) / 6 = 98.99%
```

---

## Visual Improvements

### New Warning Indicator

When features are skipped, users see a highlighted note:

```
┌─────────────────────────────────────────────────────┐
│ ⚠️ Note: 3 feature(s) not available in database    │
│    for this object                                  │
└─────────────────────────────────────────────────────┘
```

**Styling:**
- Orange background (#fff3e0)
- Orange left border (#ff9800)
- Italic text for emphasis
- Clearly visible but non-intrusive

---

## Files Modified

### 1. `web_app/app.py`

**Changes:**
- Added `actually_matched_features` list
- Track features that were successfully compared
- Add `features_provided` and `features_skipped` to match_info
- Only include compared features in `matched_features`

**Lines changed:** ~175-235

---

### 2. `web_app/static/js/app.js`

**Changes:**
- Updated label: "Features Matched" → "Features Compared"
- Added conditional note for skipped features
- Table now only shows features in `matchedFeatures` list (which contains only compared features)

**Lines changed:** ~200-220

---

### 3. `web_app/static/css/style.css`

**Changes:**
- Added `.features-skipped` class
- Orange warning styling
- Italic emphasis

**Lines added:** ~521-530

---

## Testing

### Test Case 1: All Features Available

**Input:** 9 features  
**Database:** 9 features (all available)  
**Expected:**
```
Features Compared: 9/9
[No warning note]
[Table shows all 9 features]
```

---

### Test Case 2: Some Features Missing

**Input:** 9 features  
**Database:** 6 features (3 N/A)  
**Expected:**
```
Features Compared: 6/9
Note: 3 feature(s) not available in database for this object
[Table shows only 6 features]
```

---

### Test Case 3: Minimal Features

**Input:** 3 features  
**Database:** 3 features (all available)  
**Expected:**
```
Features Compared: 3/3
[No warning note]
[Table shows 3 features]
```

---

## Why This Matters

### 1. **Transparency**
Users know exactly which features were used for matching

### 2. **Accuracy**
The count reflects reality, not assumptions

### 3. **Trust**
Clear explanation builds confidence in results

### 4. **Debugging**
Easier to understand why similarity might be lower/higher

### 5. **Data Quality Awareness**
Users learn that database coverage varies by object

---

## Impact on Similarity Scores

### Before Fix:
- Similarity calculated on 6 features
- Display shows "9/9 matched" (misleading)
- N/A rows confuse users

### After Fix:
- Similarity calculated on 6 features (same)
- Display shows "6/9 compared" (accurate)
- Clear explanation of what happened

**Key Point:** The similarity calculation was already correct (it skipped N/A values). This fix just makes the **display** match the **reality**.

---

## Summary

✅ **Accurate feature count** - Shows actual compared features  
✅ **Clear warnings** - Explains when features are skipped  
✅ **Clean tables** - No more confusing N/A rows  
✅ **Better UX** - Users understand what they're seeing  
✅ **Transparency** - Honest about data availability  

**Result:** Users now clearly understand that similarity is based on features that exist in both their input AND the database! 🎯
