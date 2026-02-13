# Pipeline Audit Map

> **Generated:** January 9, 2026
> **Auditor:** MLOps Pipeline Audit Tool
> **Purpose:** Document the complete data flow from raw production data through inference and evaluation

---

## 1. Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Raw Garmin Data │────▶│  sensor_data_    │────▶│ sensor_fused_   │
│   (Excel)       │     │  pipeline.py     │     │ 50Hz.csv        │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ production_X.npy│◀────│  preprocess_     │◀────│ Unit Detection  │
│ (Normalized     │     │  data.py         │     │ & Conversion    │
│  Windows)       │     └──────────────────┘     └─────────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Predictions CSV │◀────│  run_inference.py│◀────│ Model Loading   │
│ + Probabilities │     │                  │     │ (Keras)         │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Evaluation      │◀────│ evaluate_        │
│ Reports         │     │ predictions.py   │
└─────────────────┘     └──────────────────┘
```

---

## 2. Stage-by-Stage Breakdown

### Stage 1: Raw Data Ingestion
| Attribute | Value |
|-----------|-------|
| **Script** | `src/sensor_data_pipeline.py` |
| **Input** | Raw Excel files from Garmin watch (accelerometer.xlsx, gyroscope.xlsx) |
| **Output** | `data/processed/sensor_fused_50Hz.csv` |
| **Config** | None (hardcoded 50Hz target) |

**Key Operations:**
1. Load Excel files with sensor data
2. Parse list columns (x, y, z stored as strings)
3. Create precise timestamps
4. Merge accelerometer + gyroscope with 1ms tolerance
5. Resample to 50Hz
6. Output: CSV with columns `[timestamp_ms, timestamp_iso, Ax, Ay, Az, Gx, Gy, Gz]`

<<<<<<< HEAD
**⚠️ ISSUE FOUND:** Column headers have trailing spaces (` Ax ` not `Ax`) - FIXED by `.str.strip()` in preprocessing
=======
**✅ VERIFIED:** Column names are clean (no trailing spaces). The column renaming in `sensor_data_pipeline.py` line 494 explicitly defines column names as `{"x": "Ax", "y": "Ay", "z": "Az"}` with no spaces. CSV output uses `index=False` and standard pandas `.to_csv()` which does not add trailing spaces.
>>>>>>> 8632082 (Complete 10-stage MLOps pipeline with AdaBN domain adaptation)

---

### Stage 2: Preprocessing (Unit Conversion + Normalization + Windowing)
| Attribute | Value |
|-----------|-------|
| **Script** | `src/preprocess_data.py` |
| **Input** | CSV with sensor data (expects columns `Ax, Ay, Az, Gx, Gy, Gz`) |
| **Output** | `data/prepared/production_X.npy` (shape: n_windows, 200, 6) |
| **Output** | `data/prepared/production_metadata.json` |
| **Config Files** | `config/pipeline_config.yaml`, `data/prepared/config.json` |

**Key Operations:**
1. Detect data format (labeled vs unlabeled)
2. Detect accelerometer units (milliG vs m/s²)
3. Convert milliG → m/s² if needed (factor: 0.00981) ✅ VERIFIED CORRECT
4. Optional: Gravity removal (high-pass filter) OR Domain calibration
5. Normalize using training scaler (from `config.json`) ✅ VERIFIED CORRECT
6. Create sliding windows (200 samples, 50% overlap) ✅ VERIFIED CORRECT
7. Save as float32 NumPy array

**Config Files Read:**
- `data/prepared/config.json` → scaler_mean, scaler_scale, sensor_cols
- `config/pipeline_config.yaml` → gravity_removal, sampling_frequency_hz

**✅ VERIFIED:** Preprocessing code is correct. Column order is handled properly.

---

### Stage 3: Inference
| Attribute | Value |
|-----------|-------|
| **Script** | `src/run_inference.py` |
| **Input** | `data/prepared/production_X.npy` |
| **Output** | `data/prepared/predictions/predictions_YYYYMMDD_HHMMSS.csv` |
| **Output** | `data/prepared/predictions/*_probabilities.npy` |
| **Output** | `data/prepared/predictions/*_metadata.json` |
| **Model** | `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` |

**Key Operations:**
1. Load Keras model
2. Validate input shape: (n_windows, 200, 6)
3. Run batch inference
4. Extract predictions (argmax) and confidence scores (max prob)
5. Map class indices to activity names
6. Export CSV with columns: window_id, predicted_class, predicted_activity, confidence, etc.

**Model Contract:**
- Input shape: `(None, 200, 6)` - 200 timesteps, 6 sensors
- Output shape: `(None, 11)` - 11 activity classes
- Expected channel order: `[Ax, Ay, Az, Gx, Gy, Gz]`

---

### Stage 4: Evaluation
| Attribute | Value |
|-----------|-------|
| **Script** | `src/evaluate_predictions.py` |
| **Input** | Predictions CSV from inference |
| **Input (optional)** | Ground truth labels |
| **Output** | `outputs/evaluation/` reports |
| **Config** | None (hardcoded paths) |

**Key Operations:**
1. Load predictions CSV
2. Analyze distribution, confidence, temporal patterns (unlabeled)
3. If labels available: compute accuracy, precision, recall, F1, confusion matrix
4. Export JSON reports

---

## 3. Configuration Files

### 3.1 `data/prepared/config.json` (Training Scaler Config)
```json
{
  "window_size": 200,
  "overlap": 0.5,
  "target_hz": 50,
  "sensor_cols": ["Ax_w", "Ay_w", "Az_w", "Gx_w", "Gy_w", "Gz_w"],
  "n_classes": 11,
  "scaler_mean": [3.22, 1.28, -3.53, 0.60, 0.23, 0.09],
  "scaler_scale": [6.57, 4.35, 3.24, 49.93, 14.81, 14.17],
  "activity_to_label": {...}
}
```

**Used by:** `preprocess_data.py` (normalization)

### 3.2 `config/pipeline_config.yaml`
```yaml
preprocessing:
  enable_gravity_removal: false
  gravity_filter:
    cutoff_hz: 0.3
    order: 3
  sampling_frequency_hz: 50
  resampling:
    enabled: true
    target_hz: 50

inference:
  window_size: 200
  window_overlap: 0.5
  batch_size: 32
```

**Used by:** `preprocess_data.py` (gravity removal), `data_validator.py`

### 3.3 `config/mlflow_config.yaml`
```yaml
mlflow:
  tracking_uri: "mlruns"
  experiment_name: "anxiety-activity-recognition"
  registry:
    model_name: "har-1dcnn-bilstm"
```

**Used by:** `mlflow_tracking.py`, `run_inference.py`

### 3.4 `models/pretrained/model_info.json`
```json
{
  "input_shape": [null, 200, 6],
  "output_shape": [null, 11],
  "window_size": 200,
  "num_features": 6,
  "num_classes": 11,
  "total_params": 499131
}
```

---

## 4. Data Files and Expected Formats

### 4.1 Training Data
| File | Path | Format |
|------|------|--------|
| Labeled training | `research_papers/all_users_data_labeled.csv` | CSV |
| | Columns: `timestamp, Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w, activity, User` |

### 4.2 Production Data
| File | Path | Format |
|------|------|--------|
| Raw fused | `data/processed/sensor_fused_50Hz.csv` | CSV (milliG) |
| | Columns: `timestamp_ms, timestamp_iso, Ax, Ay, Az, Gx, Gy, Gz` |
| Labeled Garmin | `data/prepared/garmin_labeled.csv` | CSV (m/s²) |
| | Columns: `timestamp, Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w, activity, User` |
| Preprocessed | `data/prepared/production_X.npy` | NumPy (normalized) |
| | Shape: `(1815, 200, 6)` dtype: float32 |

---

## 5. Critical Pipeline Contract Requirements

### 5.1 Sensor Channel Order
The model expects EXACTLY this order:
1. Ax (accelerometer X)
2. Ay (accelerometer Y)
3. Az (accelerometer Z)
4. Gx (gyroscope X)
5. Gy (gyroscope Y)
6. Gz (gyroscope Z)

### 5.2 Units After Conversion
- Accelerometer: **m/s²** (range: ~-50 to +50)
- Gyroscope: **deg/s** (range: ~-500 to +500)

### 5.3 Normalization
Must use TRAINING scaler statistics:
- Mean: `[3.22, 1.28, -3.53, 0.60, 0.23, 0.09]`
- Scale: `[6.57, 4.35, 3.24, 49.93, 14.81, 14.17]`

After normalization, data should have:
- Mean ≈ 0 (within ±1.0)
- Std ≈ 1 (within ±0.5)

### 5.4 Window Size
- 200 samples at 50Hz = 4 seconds
- 50% overlap = 100 sample step size

---

## 6. Identified Issues (UPDATED Jan 9, 2026)

| ID | Severity | Issue | Status | Resolution |
|----|----------|-------|--------|------------|
| P1 | 🔴 CRITICAL | Production data is IDLE with no activity | **ROOT CAUSE** | Collect new data |
| P2 | ✅ RESOLVED | Column name mismatch | **NOT AN ISSUE** | Code handles correctly |
| P3 | ✅ RESOLVED | Trailing spaces | **HANDLED** | `.str.strip()` applied |
| P4 | ✅ VERIFIED | Unit conversion | **CORRECT** | 0.00981 verified |
| P5 | ✅ VERIFIED | Normalization | **CORRECT** | Uses training scaler |

**Root Cause:** Low std (~0.02) is correct because input has no activity variance.

---

## 7. Data Flow Validation Checklist

- [x] Raw data columns match expected schema ✅
- [x] Timestamp is monotonic ✅
- [x] Sampling rate is 50Hz ✅
- [x] Unit conversion (milliG → m/s²) ✅ VERIFIED
- [x] Channel order preserved ✅ VERIFIED
- [x] Scaler applied correctly ✅ VERIFIED
- [x] Window shape (200, 6) ✅ VERIFIED
- [ ] **PENDING:** Input data contains activities
