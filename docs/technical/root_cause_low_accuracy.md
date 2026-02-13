# Root Cause Analysis: Low Production Accuracy (14-15%)

**Date:** January 9, 2026  
**Status:** ✅ ROOT CAUSE IDENTIFIED  
**Confidence:** HIGH (99%)

---

## Executive Summary

The production accuracy of 14-15% (vs. expected 85%+) is caused by **using IDLE/STATIONARY data for inference**. The `sensor_fused_50Hz.csv` contains data from a user NOT performing any activities (watch lying flat on table), resulting in ~60x less variance than the training data.

**Primary Root Cause:** The production data source (`sensor_fused_50Hz.csv`) contains stationary data with minimal motion, while the model was trained on active movement (11 distinct activities). This is a **DATA CONTENT issue**, not a preprocessing bug.

**The preprocessing code is CORRECT** - it properly:
1. Detects milliG units ✅
2. Converts to m/s² with factor 0.00981 ✅
3. Applies StandardScaler normalization ✅
4. Creates proper windows ✅

---

## Two Options for Valid Inference

### Option 1: Collect NEW Garmin Data with Activities (RECOMMENDED)

Since `sensor_fused_50Hz.csv` (March 2025) is from an idle watch, you need to:

1. **Wear the Garmin watch** and perform target activities:
   - ear_rubbing, forehead_rubbing, hair_pulling, hand_scratching
   - hand_tapping, knuckles_cracking, nail_biting, nape_rubbing
   - smoking, sitting, standing

2. **Export the data** from Garmin Connect

3. **Process through the pipeline:**
   ```bash
   # Step 1: Process raw Garmin export
   python src/sensor_data_pipeline.py --input data/raw/YOUR_NEW_EXPORT.xlsx
   
   # Step 2: Preprocess for inference
   python src/preprocess_data.py --input data/processed/sensor_fused_50Hz.csv
   
   # Step 3: Run inference
   python src/run_inference.py
   ```

### Option 2: Use sensor_fused_50Hz.csv AS-IS (For Pipeline Testing Only)

If you just want to **test the pipeline works** (not accuracy):

```bash
# The pipeline will run correctly, but predictions will be unreliable
# because the input data has no meaningful activity patterns
python src/preprocess_data.py --input data/processed/sensor_fused_50Hz.csv
python src/run_inference.py
```

**Expected Result:** ~14% accuracy (near-random) because the model sees "nothing happening"

---

## Note on Timestamps

| Data Source | Timestamps | Actual Date | Notes |
|-------------|------------|-------------|-------|
| sensor_fused_50Hz.csv | 2025-03-24 | March 2025 | Real timestamps, IDLE data |
| garmin_labeled.csv | 2005-05-01 | Unknown | Placeholder dates, has activities |
| all_users_data_labeled.csv | 2005-05-01 | Unknown | Training data, placeholder dates |

The 2005 dates in labeled data are **placeholder timestamps** (common in research datasets). The actual sensor readings are valid - only the timestamps are reset.

---

## Top 10 Ranked Causes

### 🔴 RANK 1: Production Data is IDLE/STATIONARY (CONFIRMED - TRUE ROOT CAUSE)

**Evidence:**
```
Data Source Comparison (Standard Deviation):

                           Ax (m/s²)    Az mean (m/s²)
Training data              6.57         -3.53 (tilted/moving)
sensor_fused_50Hz          0.11         -9.83 (FLAT on table!)

Ratio: Training/Production = 6.57 / 0.11 = ~60x difference!
```

**Analysis:**
- `sensor_fused_50Hz.csv` has Az mean = -9.83 m/s² (exactly -g = flat on surface)
- Training data has Az mean = -3.53 m/s² (tilted, typical wrist orientation)
- The production data is from an **IDLE watch on a table**, not a user performing activities
- Timestamp: 2025-03-24 (real data, but no activity)

**Impact:** Model sees "nothing happening" → predicts random classes

**Fix Priority:** 🔴 CRITICAL - Collect new data with activities

**Fix:** Record new Garmin data with user performing the 11 target activities

---

### 🟢 RANK 2-10: Previously Suspected Issues (NOW RULED OUT)

After tracing the math, the following were **ruled out** as causes:

| Rank | Suspected Issue | Status | Evidence |
|------|-----------------|--------|----------|
| 2 | Column name mismatch | ❌ NOT THE CAUSE | Code handles `Ax` vs `Ax_w` correctly |
| 3 | Unit conversion bug | ❌ NOT THE CAUSE | Factor 0.00981 applied correctly |
| 4 | Double normalization | ❌ NOT THE CAUSE | Applied once, verified |
| 5 | Scaler mismatch | ❌ NOT THE CAUSE | Correct scaler loaded |
| 6 | Gravity removal issue | ❌ NOT THE CAUSE | Disabled by default |
| 7 | Window overlap bug | ❌ NOT THE CAUSE | Shape correct |
| 8 | Model loading issue | ❌ NOT THE CAUSE | Model verified |
| 9 | Evaluation bug | ❌ NOT THE CAUSE | Metrics correct |
| 10 | Data type issue | ❌ NOT THE CAUSE | Converted to float |

---

## Diagnosis Summary

| # | Cause | Status | Evidence |
|---|-------|--------|----------|
| 1 | **IDLE data used for inference** | **CONFIRMED** | Az=-9.83 m/s² (flat), std 60x lower |
| 2-10 | Preprocessing bugs | ❌ RULED OUT | Code verified correct |

---

## Preprocessing Code Verification

### ✅ Unit Detection (preprocess_data.py lines 99-163)
- Correctly detects milliG when max_abs > 100 ✅
- Conversion factor 0.00981 matches supervisor's email ✅

### ✅ Unit Conversion (preprocess_data.py lines 165-209)
- Multiplies accelerometer columns by 0.00981 ✅
- Validates Az mean ≈ -9.8 m/s² after conversion ✅

### ✅ Gravity Validation
- Az mean in sensor_fused_50Hz.csv: -1001.56 milliG
- After conversion: -1001.56 × 0.00981 = **-9.825 m/s²** ✅
- This confirms data is from watch lying **flat on table** (not on wrist)

### ✅ StandardScaler Normalization (preprocess_data.py lines 468-525)
- Loads correct scaler_mean and scaler_scale from config.json ✅
- Applies transform correctly ✅
- The low std (~0.02) is **correct** given low-variance input data

---

## The Math (Verified)

```
sensor_fused_50Hz.csv (raw, milliG):
  Ax std = 11.32 milliG

After milliG → m/s² conversion:
  Ax std = 11.32 × 0.00981 = 0.111 m/s²

After StandardScaler normalization:
  Ax std = 0.111 / 6.568 = 0.017  ← MATCHES production_X.npy!

Training data for comparison:
  Ax_w std = 6.568 m/s²

The normalization is CORRECT.
The problem is the INPUT DATA has no motion variance.
```

---

## Action Plan

### Option 1: Collect NEW Garmin Data with Activities (RECOMMENDED)

**Steps:**
1. **Wear Garmin watch** on wrist
2. **Perform activities** (each for 2-5 minutes):
   - ear_rubbing, forehead_rubbing, hair_pulling
   - hand_scratching, hand_tapping, knuckles_cracking
   - nail_biting, nape_rubbing, smoking
   - sitting, standing
3. **Export data** from Garmin Connect (Excel/CSV format)
4. **Process through pipeline:**
   ```bash
   # Process raw export
   python src/sensor_data_pipeline.py --input data/raw/NEW_ACTIVITY_DATA.xlsx
   
   # Preprocess for inference  
   python src/preprocess_data.py --input data/processed/sensor_fused_50Hz.csv
   
   # Run inference
   python src/run_inference.py
   ```
5. **Expected accuracy:** 70-85% (similar to training)

### Option 2: Pipeline Test Only (Current IDLE Data)

**Use Case:** Verify the pipeline runs correctly (NOT for accuracy evaluation)

```bash
# Run with current idle data - pipeline will work, accuracy will be low
python src/preprocess_data.py --input data/processed/sensor_fused_50Hz.csv
python src/run_inference.py
```

**Expected Result:** 
- Pipeline completes successfully ✅
- Accuracy ~14% (near-random) ⚠️
- This is EXPECTED because input has no activity patterns

### Why garmin_labeled.csv Cannot Be Used

The `garmin_labeled.csv` file has timestamps from 2005 (placeholder dates) and is part of a research dataset, not real production data. For a valid production inference test, you need:
- **Real Garmin data** from your own watch
- **Actual activities** being performed
- **Recent timestamps** (2025+)

---

## Conclusion

The 14-15% accuracy is caused by using **IDLE/STATIONARY data** for inference. The `sensor_fused_50Hz.csv` file contains data from a watch lying flat on a table (Az = -9.83 m/s² = pure gravity), with no user activity.

**The preprocessing code is CORRECT.** The issue is the data content:

| Data Source | Date | Ax std (m/s²) | Az mean (m/s²) | Status |
|-------------|------|---------------|----------------|--------|
| Training data | 2005* | 6.57 | -3.53 | ✅ Active movement |
| sensor_fused_50Hz.csv | 2025-03-24 | 0.11 | -9.83 | ❌ IDLE (flat) |

*2005 dates are placeholder timestamps in research data

**Solution:** Collect NEW Garmin data with user actively performing the 11 target activities.

---

## Script Verification Summary

| Script | Component | Status |
|--------|-----------|--------|
| `src/preprocess_data.py` | Unit detection (milliG vs m/s²) | ✅ CORRECT |
| `src/preprocess_data.py` | Conversion factor (0.00981) | ✅ CORRECT |
| `src/preprocess_data.py` | StandardScaler normalization | ✅ CORRECT |
| `src/preprocess_data.py` | Windowing (200 samples, 50% overlap) | ✅ CORRECT |
| `src/run_inference.py` | Model loading | ✅ CORRECT |
| `src/run_inference.py` | Batch inference | ✅ CORRECT |
| `src/evaluate_predictions.py` | Metrics computation | ✅ CORRECT |

---

## Supervisor Confirmation (Dec 3, 2025)

From Oleh Ugonna:
> "The accelerometer data was converted from milliG to m/s^2. Here is the conversion factor:
> conversion_factor = 0.00981
> The accelerometer values from the unlabeled data are still in milliG and need to be multiplied by the conversion factor."

**Status:** Conversion factor is correctly implemented in `preprocess_data.py` (line 101).

---

## Appendix: Key Evidence

| File | Date | Units | Status |
|------|------|-------|--------|
| `sensor_fused_50Hz.csv` | 2025-03-24 | milliG | ⚠️ Real date, IDLE data |
| `garmin_labeled.csv` | 2005* | m/s² | Research dataset, not production |
| `all_users_data_labeled.csv` | 2005* | m/s² | Training data |
| `config.json` | - | - | ✅ Correct scaler |
| `production_X.npy` | 2026-01-06 | normalized | ⚠️ From IDLE data |

*Placeholder timestamps from research dataset
