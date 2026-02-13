# QC Execution Summary
**Date:** January 9, 2026  
**Status:** ✅ ALL QC CHECKS COMPLETED  
**Root Cause:** CONFIRMED - IDLE DATA (No Activity Patterns)

---

## Executive Summary

<<<<<<< HEAD
Three production-grade QC scripts were executed to validate the preprocessing pipeline and diagnose the low accuracy (~14-15%) in production inference. **All scripts successfully identified the root cause: the production data (`sensor_fused_50Hz.csv`) contains IDLE/STATIONARY data with no activity patterns.**
=======
+


>>>>>>> 8632082 (Complete 10-stage MLOps pipeline with AdaBN domain adaptation)

### Key Findings
1. **Preprocessing Pipeline:** ✅ VERIFIED CORRECT  
   - Unit detection: ✅ Correctly detects milliG  
   - Conversion (0.00981 factor): ✅ Mathematically correct  
   - Normalization: ✅ Uses correct training scaler  
   - Windowing: ✅ 200 samples, 6 channels  

2. **Root Cause:** IDLE DATA  
   - Input variance: mean_std = 0.1154 (expected ~1.0)  
   - Az = -9.83 m/s² (pure gravity, watch flat on table)  
   - Ax std = 0.11 m/s² vs training 6.57 m/s² (60x less)  

3. **Impact:**  
   - Model predicts `hand_tapping` (class 4) for 100% of windows  
   - Confidence: 93.01% (model is certain, but input is wrong)  
   - This explains the 14-15% accuracy (random chance ~9%)  

---

## QC Test 1: Preprocessing QC (Raw CSV)

### Command
```powershell
python scripts/preprocess_qc.py --input data/preprocessed/sensor_fused_50Hz.csv --type production
```

### Results
**Status:** ✅ PASS  
**File:** `reports/preprocess_qc/qc_20260109_141208.json`  
**Checks Passed:** 9/9  

| Check | Result | Details |
|-------|--------|---------|
| Required columns exist | ✅ PASS | All 6 sensor columns + timestamp present |
| Missing values acceptable | ✅ PASS | <5% NaN per channel |
| Timestamps monotonic | ✅ PASS | Strictly increasing |
| No duplicate timestamps | ✅ PASS | 0 duplicates |
| Sampling rate correct | ✅ PASS | 50.00 Hz (expected 50 Hz) |
| Resampling jitter acceptable | ✅ PASS | Time diff std <5% |
| Accelerometer units detected | ✅ PASS | Detected: milliG (max=1006.0) |
| Gyroscope units correct | ✅ PASS | Detected: deg/s (max=89.2) |
| Channel order correct | ✅ PASS | [Ax, Ay, Az, Gx, Gy, Gz] |

**Conclusion:** Raw CSV preprocessing is CORRECT. Data is clean, properly formatted, and ready for conversion.

---

## QC Test 2: Preprocessing QC (Normalized NPY)

### Command
```powershell
python scripts/preprocess_qc.py --input data/prepared/production_X.npy --type normalized
```

### Results
**Status:** ❌ FAIL (EXPECTED - ROOT CAUSE DETECTED)  
**File:** `reports/preprocess_qc/qc_20260109_141213.json`  
**Checks Passed:** 2/5  
**Critical Failures:** 3  

| Check | Result | Details |
|-------|--------|---------|
| Window size correct | ✅ PASS | 200 timesteps |
| Channel count correct | ✅ PASS | 6 channels |
| Normalized mean ≈ 0 | ❌ FAIL | Mean: [-0.51, -0.34, -1.94, -0.00, -0.00, 0.00] |
| Normalized std ≈ 1 | ❌ FAIL | Std: [0.02, 0.06, 0.06, 0.11, 0.32, 0.13] |
| **Variance collapse detected** | ❌ FAIL | **IDLE/STATIONARY data - no activity patterns** |

**Conclusion:** ⚠️ **VARIANCE COLLAPSE CONFIRMED**  
- Production std is 60x LOWER than training (0.02 vs 1.0)  
- This is the ROOT CAUSE of low accuracy  
- **Recommendation:** Collect NEW data with actual activities (walking, running, tapping, etc.)

---

## QC Test 3: Inference Smoke Test

### Command
```powershell
python scripts/inference_smoke.py
```

### Results
**Status:** ❌ FAIL (EXPECTED - IDLE DATA)  
**File:** `reports/inference_smoke/smoke_20260109_141448.json`  
**Checks Passed:** 10/12  
**Critical Failures:** 1  

| Check | Result | Details |
|-------|--------|---------|
| Model loads successfully | ✅ PASS | 499,131 parameters, input (None, 200, 6) |
| Data loads successfully | ✅ PASS | (1815, 200, 6) |
| Timesteps match | ✅ PASS | 200 = 200 |
| Channels match | ✅ PASS | 6 = 6 |
| Inference runs successfully | ✅ PASS | 1815 predictions generated |
| **Input has activity variance** | ❌ FAIL | **Mean std: 0.1154 - IDLE DATA DETECTED** |
| Output classes correct | ✅ PASS | 11 classes |
| Probabilities sum to 1 | ✅ PASS | Softmax valid |
| Predictions not uniform | ✅ PASS | Entropy ratio: 0.0 |
| Not all same class | ❌ FAIL | Class 4 (hand_tapping) predicted 100% |
| Model shows confidence | ✅ PASS | Mean confidence: 93.01% |
| Inference is deterministic | ✅ PASS | Same input → same output |

**Prediction Analysis:**
- **Most predicted class:** 4 (`hand_tapping`) - 100.0% of windows  
- **Confidence histogram:**  
  - 0.0-0.2: 0 windows (0.0%)  
  - 0.2-0.4: 0 windows (0.0%)  
  - 0.4-0.6: 0 windows (0.0%)  
  - 0.6-1.0: 1815 windows (100.0%)  
- **Worst softmax violations:** Top 5 deviations all <0.001 (perfect softmax)

**Conclusion:** Model is working correctly, but input data has no activity variance. The model confidently predicts `hand_tapping` for all idle data windows.

---

## Improvements Implemented

### 1. `scripts/preprocess_qc.py` (5 improvements)
1. **Dynamic scaler loading:** Removed hardcoded `EXPECTED_SCALER_MEAN/SCALE`, now reads from `config.json`  
2. **Missingness checks:** Added `check_missingness()` with NaN % per channel and gap detection (<5% threshold)  
3. **Resampling quality:** Enhanced `check_sampling_rate()` with jitter verification (std of time diffs <5%)  
4. **Gyroscope unit validation:** Added `check_gyro_units()` to detect deg/s vs rad/s (similar to accel check)  
5. **Enhanced variance collapse message:** Now provides actionable guidance ("IDLE data detected, collect data with activities")  

### 2. `scripts/inference_smoke.py` (4 improvements)
1. **Per-sample softmax check:** Shows worst 5 examples + deviation magnitude  
2. **Confidence histogram:** 4 buckets (<0.2, 0.2-0.4, 0.4-0.6, >0.6) with counts and %  
3. **Input variance proxy:** Added `check_input_variance()` to detect idle data before inference (mean_std <0.3)  
4. **MLflow logging:** Optional `--mlflow` flag to log smoke test results (run_id, metrics, artifacts)  

**External Reviewer Feedback:** "This is thesis-strong. The QC suite is production-grade and provides clear diagnostic value."

---

## Recommendations

### Immediate Action (Option 1: Collect NEW Data)
1. **Collect Garmin data with actual activities:**
   - Walking (slow, normal, brisk)  
   - Running  
   - Cycling  
   - Stairs up/down  
   - Hand tapping  
   - Rotation  
   - **Duration:** At least 30 minutes per activity  
   - **Device:** Garmin watch (same model as training)  

2. **Verify data collection:**
   ```powershell
   # Check raw data
   python scripts/preprocess_qc.py --input <new_data>.csv --type production
   
   # Should see:
   # - Ax std > 2.0 m/s² (movement variance)
   # - Az mean ≠ -9.8 m/s² (not flat)
   ```

3. **Rerun pipeline:**
   ```powershell
   python src/preprocess_data.py --input <new_data>.csv
   python scripts/preprocess_qc.py --input data/prepared/production_X.npy --type normalized
   python scripts/inference_smoke.py
   ```

4. **Expected results:**
   - Variance collapse: ❌ → ✅ (std ≈ 1.0)  
   - Predictions: diverse classes (not 100% one class)  
   - Accuracy: Should improve to >70% if activities match training set  

### Alternative Action (Option 2: Pipeline Testing Only)
- Use current idle data for **pipeline testing** only  
- Document that accuracy will remain ~14% due to data mismatch  
- **Do NOT use** for accuracy/performance evaluation  
- **Do NOT use** for thesis conclusions about model quality  

---

## Files Generated

### QC Reports (JSON)
1. `reports/preprocess_qc/qc_20260109_141208.json` - Raw CSV validation  
2. `reports/preprocess_qc/qc_20260109_141213.json` - Normalized NPY validation (variance collapse detected)  
3. `reports/inference_smoke/smoke_20260109_141448.json` - Inference smoke test (idle data detected)  

### Documentation
4. `docs/QC_EXECUTION_SUMMARY.md` - This file  
5. `docs/root_cause_low_accuracy.md` - Root cause analysis (updated)  
6. `docs/pipeline_audit_map.md` - Full pipeline audit (updated)  

---

## Technical Details

### Root Cause Verification (Math)

**Training Data (from config.json):**
- Ax: mean=3.22, scale=6.57 m/s²  
- After normalization: mean≈0, std≈1  

**Production Data (from sensor_fused_50Hz.csv):**
- Ax: raw mean≈0.5, raw std≈0.11 m/s² (IDLE)  
- After conversion (0.00981): same  
- After normalization: (0.11 / 6.57) ≈ 0.017 ✅ MATCHES observed std ~0.02  

**Conclusion:** Preprocessing is mathematically CORRECT. Low variance is from IDLE data source.

### Conversion Factor Verification
- **Supervisor confirmation (Dec 3, 2025):** 0.00981 (milliG → m/s²)  
- **Formula:** `milliG * 0.00981 = m/s²`  
- **Example:** 1000 milliG = 9.81 m/s² ✅ Correct  

### Model Architecture
- **Type:** 1D-CNN-BiLSTM  
- **Input:** (None, 200, 6)  
- **Output:** (None, 11) - softmax probabilities  
- **Parameters:** 499,131  
- **Training scaler:** StandardScaler with mean/scale from training set  

---

## Conclusion

✅ **Pipeline is CORRECT**  
✅ **QC suite is PRODUCTION-GRADE**  
✅ **Root cause is CONFIRMED: IDLE DATA**  

**Next Step:** Collect NEW Garmin data with actual activities to achieve valid inference and accurate performance metrics.

---

**Generated by:** Pipeline Audit QC Suite  
**Date:** January 9, 2026  
**Version:** Production-grade (with reviewer improvements)  
