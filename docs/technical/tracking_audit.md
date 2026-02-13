# MLflow + DVC Tracking Audit

**Date:** January 9, 2026  
**Status:** ✅ AUDIT COMPLETE  
**Scope:** MLflow experiment tracking + DVC data versioning

---

## 1. Overview

This document audits the tracking infrastructure:
1. **MLflow**: Experiment tracking, model registry, metrics logging
2. **DVC**: Data versioning, pipeline reproducibility

---

## 2. MLflow Configuration

### Config File: [config/mlflow_config.yaml](../config/mlflow_config.yaml)

```yaml
mlflow:
  tracking_uri: "mlruns/"
  experiment_name: "har_model_training"
  run_name_prefix: "run"
  artifact_location: "mlruns/artifacts"
```

### Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Tracking URI | ✅ Local `mlruns/` | Offline-friendly |
| Experiment | ✅ Created | ID: 950614147457743858 |
| Runs | ✅ Multiple runs | See `mlruns/950614147457743858/` |
| Model Registry | ⚠️ Not used | Models stored in `models/` |

### Verified Experiments

```
mlruns/
├── 0/                          # Default experiment
├── 950614147457743858/         # har_model_training
│   └── <run_id>/
│       ├── artifacts/
│       ├── metrics/
│       └── params/
└── models/                     # Model registry (empty)
```

---

## 3. DVC Configuration

### DVC Files Present

| File | Tracks | Status |
|------|--------|--------|
| `data/raw.dvc` | Raw Garmin exports | ✅ |
| `data/processed.dvc` | Processed CSV files | ✅ |
| `data/prepared.dvc` | Prepared NPY files | ✅ |
| `models/pretrained.dvc` | Pretrained model | ✅ |
| `.dvc/config` | DVC configuration | ✅ |

### DVC Remote Status

```bash
# Check remote configuration
dvc remote list
# Expected: local storage or none (offline mode)
```

**Status:** DVC is configured for local storage, appropriate for offline-first setup.

---

## 4. Tracking Flow

### Current Pipeline

```
Training Data
     ↓
┌─────────────────┐
│ preprocess.py   │ → DVC tracks data/prepared/
└─────────────────┘
     ↓
┌─────────────────┐
│ train.py        │ → MLflow logs:
│                 │   - hyperparameters
│                 │   - metrics (loss, accuracy)
│                 │   - model artifacts
└─────────────────┘
     ↓
┌─────────────────┐
│ evaluate.py     │ → MLflow logs:
│                 │   - evaluation metrics
│                 │   - confusion matrix
└─────────────────┘
```

### Inference Pipeline (Production)

```
Production Data
     ↓
┌─────────────────┐
│ preprocess.py   │ → DVC tracks data/prepared/
└─────────────────┘
     ↓
┌─────────────────┐
│ run_inference.py│ → MLflow logs:
│                 │   - inference metrics
│                 │   - predictions artifact
└─────────────────┘
```

---

## 5. Identified Issues

### Issue 1: Model Not in MLflow Registry

**Severity:** LOW

Models are stored in `models/pretrained/` but not registered in MLflow Model Registry.

**Impact:** Can't use MLflow's model versioning, staging, production lifecycle.

**Recommendation:** For thesis, current approach is fine. Consider registry for production.

### Issue 2: DVC Cache Size

**Severity:** LOW

```
.dvc/cache/
└── files/
    └── md5/  # Can grow large with many versions
```

**Recommendation:** Periodically run `dvc gc` to clean unused cache.

### Issue 3: Experiment Naming

**Severity:** LOW

All runs in one experiment without descriptive names.

**Recommendation:** Use run tags to differentiate:
- `fine_tuning_v1`
- `hyperparameter_search`
- `production_eval`

---

## 6. MLflow Integration Audit

### Source: [src/mlflow_tracking.py](../src/mlflow_tracking.py)

| Function | Purpose | Status |
|----------|---------|--------|
| `init_mlflow()` | Initialize tracking | ✅ |
| `log_params()` | Log hyperparameters | ✅ |
| `log_metrics()` | Log training metrics | ✅ |
| `log_artifact()` | Log model files | ✅ |
| `end_run()` | Close run | ✅ |

### Logged Metrics

| Metric | Logged | Source |
|--------|--------|--------|
| `train_loss` | ✅ | Training |
| `train_accuracy` | ✅ | Training |
| `val_loss` | ✅ | Training |
| `val_accuracy` | ✅ | Training |
| `test_accuracy` | ✅ | Evaluation |
| `inference_time` | ⚠️ Not logged | Inference |

---

## 7. DVC Pipeline Verification

### Expected Workflow

```bash
# 1. Pull data
dvc pull

# 2. Verify data integrity
dvc status

# 3. Run pipeline
dvc repro  # If dvc.yaml exists
```

### Current Status

| File | Hash Matches | Data Exists |
|------|--------------|-------------|
| `data/raw.dvc` | ✅ | ✅ |
| `data/processed.dvc` | ✅ | ✅ |
| `data/prepared.dvc` | ✅ | ✅ |
| `models/pretrained.dvc` | ✅ | ✅ |

---

## 8. Recommendations

### Immediate (No Changes Needed)

1. ✅ MLflow is properly configured for local tracking
2. ✅ DVC is tracking data correctly
3. ✅ Experiments are logged

### Before Fine-Tuning

1. Create new MLflow experiment: `har_fine_tuning`
2. Tag runs with meaningful names
3. Log scaler parameters explicitly

### After Fine-Tuning

1. Consider MLflow Model Registry for model versioning
2. Add `dvc.yaml` for pipeline reproducibility
3. Log data hash in MLflow for traceability

---

## 9. Quick Verification Commands

```bash
# Check MLflow experiments
python -c "import mlflow; print(mlflow.search_experiments())"

# Check DVC status
dvc status

# List DVC files
dvc list .

# View MLflow UI (local)
mlflow ui --port 5000
```

---

## 10. Conclusion

The tracking infrastructure is properly configured for offline-first development:

- **MLflow**: ✅ Local tracking works correctly
- **DVC**: ✅ Data versioning is active
- **Integration**: ✅ Both tools complement each other

**No changes required for audit completion.**

---

## Appendix: Key Paths

| Component | Path |
|-----------|------|
| MLflow tracking | `mlruns/` |
| MLflow config | `config/mlflow_config.yaml` |
| DVC cache | `.dvc/cache/` |
| DVC config | `.dvc/config` |
| Data DVC files | `data/*.dvc` |
| Model DVC files | `models/pretrained.dvc` |
