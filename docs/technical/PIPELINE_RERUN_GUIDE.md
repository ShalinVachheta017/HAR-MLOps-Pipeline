# Pipeline Re-run Guide (Raw ➜ Validation ➜ Preprocessing ➜ Training ➜ Inference ➜ Evaluation)

> **📝 Summary:** Complete step-by-step guide to run the entire pipeline from raw data to evaluation. Contains: fresh start cleanup commands, DVC versioning steps, preprocessing commands, training/inference execution, Docker API usage, and MLflow tracking. **Use this as your main reference for running the pipeline.**

End-to-end checklist to take new raw sensor files through the full pipeline, with DVC versioning, MLflow tracking, and Docker usage.

---

## ⚡ Quick Fresh Start (Delete All Old Records)

If you want to **start from zero** and delete all old generated files:

```powershell
# Clean everything in one command
.\scripts\complete_fresh_start.ps1

# Or manually (see FRESH_START_CLEANUP_GUIDE.md for details)
Remove-Item -Path "outputs/evaluation/*.json", "outputs/evaluation/*.txt", `
                    "logs/*/*.log", "data/prepared/*.npy", "mlruns", `
                    "data/preprocessed/*.csv" -Force -ErrorAction SilentlyContinue
```

**What gets deleted (SAFE):**
- ❌ Old evaluation reports (8 files)
- ❌ Old logs (preprocessing, training, inference, evaluation)
- ❌ Old predictions (.csv, .json, .npy files)
- ❌ Old preprocessed data (sensor CSVs)
- ❌ Old prepared arrays (windowed .npy files)
- ❌ MLflow experiment history

**What stays (SAFE - won't delete):**
- ✅ Raw data (data/raw/*.xlsx)
- ✅ Pretrained model (models/pretrained/*.keras)
- ✅ Reference datasets (research_papers/*.csv)
- ✅ Code & Git repository

**For complete details:** See [FRESH_START_CLEANUP_GUIDE.md](FRESH_START_CLEANUP_GUIDE.md)

---

## Prerequisites
- Activate your env and install deps:  
  ```powershell
  conda activate thesis-mlops  # or your env
  pip install -r config/requirements.txt
  ```
- Pull tracked data/models if needed: `dvc pull`
- Set MLflow URI (local folder or server):  
  ```powershell
  $env:MLFLOW_TRACKING_URI="mlruns"   # local backend (default)
  # or when MLflow server is running:
  # $env:MLFLOW_TRACKING_URI="http://localhost:5000"
  ```

## High-level Workflow
1) Locate/raw drop → 2) Data validation → 3) Sensor fusion/resample → 4) Unit conversion + windowing → 5) DVC versioning → 6) Training (optional) → 7) Inference → 8) Evaluation → 9) Track everything in MLflow → 10) Manage Docker services/images.

## 1) Locate/Drop Raw Files
- Place accelerometer + gyroscope Excel files into `data/raw/`.
- Quick scan to confirm names:  
  ```powershell
  Get-ChildItem data/raw -File -Filter *.xlsx
  ```
- If pulling from DVC remote: `dvc pull data/raw.dvc`

## 2) Data Validation (schema + sanity)
- Run validator on a raw file (adjust filenames):  
  ```powershell
  python - <<'PY'
  from pathlib import Path
  import pandas as pd
  from src.data_validator import DataValidator

  raw_dir = Path("data/raw")
  accel = raw_dir/"2025-03-23-15-23-10-accelerometer_data.xlsx"
  gyro = raw_dir/"2025-03-23-15-23-10-gyroscope_data.xlsx"
  validator = DataValidator()

  for path in (accel, gyro):
      df = pd.read_excel(path)
      result = validator.validate(df)
      print(f"[{path.name}] valid={result.is_valid}")
      if not result.is_valid:
          print("  errors:", result.errors)
  PY
  ```
- Only proceed once both files pass validation.

## 3) Sensor Pipeline (fusion, resample, gravity toggle)
- Command:  
  ```powershell
  python src/sensor_data_pipeline.py
  ```
- What it does: merges accel/gyro, cleans, resamples to 50 Hz, optional gravity removal (`config/pipeline_config.yaml`), writes fused CSV + metadata.
- Key outputs: `pre_processed_data/sensor_fused_50Hz.csv`, `pre_processed_data/sensor_fused_meta.json`

## 4) Conversion + Windowing (model-ready arrays)
- Command:  
  ```powershell
  python src/preprocess_data.py --input pre_processed_data/sensor_fused_50Hz.csv
  ```
- What it does: auto unit conversion (milliG ↔ m/s^2), normalization with saved scaler, builds 200x6 windows (50% overlap).
- Outputs: `data/prepared/production_X.npy`, `data/prepared/production_metadata.json`

## 5) Version Data with DVC

### What Gets Versioned by DVC

DVC tracks large files and folders. Your project currently has 6 tracked items:

```
data/raw.dvc                    ← Raw sensor Excel files (~60MB)
data/processed.dvc              ← Fused CSV files (~110MB)  
data/prepared.dvc               ← Windowed arrays + predictions (~50MB)
models/pretrained.dvc           ← Pre-trained model (~18MB)
research_papers/anxiety_dataset.csv.dvc           ← Reference data (~50MB)
research_papers/all_users_data_labeled.csv.dvc    ← Training data (~70MB)
```

### Starting Fresh - DVC Workflow

**Option A: Keep everything in DVC (Recommended for thesis)**

If you want to track all new outputs with DVC:

```powershell
# After preprocessing, add new data
dvc add data/preprocessed/sensor_fused_50Hz.csv
dvc add data/prepared/production_X.npy
dvc push

# Commit DVC files to Git
git add data/preprocessed.dvc data/prepared.dvc
git add .gitignore
git commit -m "Add new pipeline run outputs to DVC"
git push
```

**Option B: Keep old data, only track final model**

If you want to skip versioning intermediate outputs:

```powershell
# Just delete the outputs without DVC tracking
Remove-Item -Path "data/preprocessed/*", "data/prepared/*.npy" -Force

# Only version final trained model (if you train)
dvc add models/trained/new_model.keras
dvc push
git add models/trained.dvc
git commit -m "Add new trained model"
```

**Option C: Keep nothing but raw data (Clean slate)**

If you want complete fresh start:

```powershell
# Delete all generated outputs from previous runs
.\scripts\complete_fresh_start.ps1

# When ready to save new results, add them:
dvc add data/prepared
dvc push
git add data/prepared.dvc
git commit -m "Fresh run: new predictions"
```

### Understanding DVC Storage

**Local Cache (.dvc/cache/):**
- Temporary storage of files you've added
- Automatically deleted when not needed
- Can be cleaned with: `dvc gc --workspace`

**Remote Storage (Google Drive / S3 / Azure):**
- Where actual files live for backup & sharing
- Configured in: `.dvc/config`
- Push to remote: `dvc push`
- Pull from remote: `dvc pull`

**Quick DVC Commands:**

```powershell
# Check status
dvc status

# See what's tracked
dvc dag

# Clean local cache
dvc gc --workspace

# Push to remote
dvc push

# Pull from remote
dvc pull

# Track new files
dvc add <path>

# Untrack files
dvc remove <path>.dvc
```

---

## 6) Training (optional, logs to MLflow)
- Local:  
  ```powershell
  python src/train.py --experiment "anxiety-activity-recognition"
  ```
- Docker (on-demand profile):  
  ```powershell
  docker-compose --profile training run --rm training python src/train.py --experiment "anxiety-activity-recognition"
  ```
- Artifacts: logged to MLflow (`mlruns/` or remote server) and models in `models/trained/` (DVC-track if keeping).

## 7) Inference

### Option A: Batch Inference (Python Script)
- Command:  
  ```powershell
  python src/run_inference.py `
    --input data/prepared/production_X.npy `
    --model models/pretrained/fine_tuned_model_1dcnnbilstm.keras `
    --output data/prepared/predictions
  ```
- Outputs: `data/prepared/predictions/predictions.csv`, `predictions_summary.json`, logs in `logs/inference/`.

### Option B: REST API Inference (Docker)

**Start the API Service:**
```powershell
# Start the inference API container
docker-compose up -d inference

# Check if service is running
docker-compose ps

# View API logs
docker-compose logs -f inference
```

**Access the API:**
- **Swagger UI (Interactive):** Open `http://localhost:8000/docs` in browser
- **ReDoc (Documentation):** Open `http://localhost:8000/redoc`
- **Health Check:** `curl http://localhost:8000/health` or visit in browser

**API Endpoints:**

1. **Health Check**
   ```powershell
   curl http://localhost:8000/health
   ```
   Response: `{"status": "healthy", "model_loaded": true}`

2. **Model Info**
   ```powershell
   curl http://localhost:8000/model/info
   ```
   Response:
   ```json
   {
     "model_name": "1D-CNN-BiLSTM",
     "input_shape": [200, 6],
     "output_classes": 11,
     "parameters": 499131
   }
   ```

3. **Single Prediction**
   ```powershell
   curl -X POST http://localhost:8000/predict `
     -H "Content-Type: application/json" `
     -d '{
       "window": [
         [0.1, 0.2, -3.5, 0.01, 0.02, 0.03],
         [0.1, 0.2, -3.5, 0.01, 0.02, 0.03],
         ...
       ]
     }'
   ```
   Response:
   ```json
   {
     "activity": "forehead_rubbing",
     "confidence": 0.497,
     "confidence_level": "UNCERTAIN",
     "probabilities": {
       "forehead_rubbing": 0.497,
       "nape_rubbing": 0.371,
       "sitting": 0.087
     },
     "inference_time_ms": 1.35
   }
   ```

4. **Batch Prediction (Multiple Windows)**
   ```powershell
   curl -X POST http://localhost:8000/predict/batch `
     -H "Content-Type: application/json" `
     -d '{
       "windows": [
         [[0.1, 0.2, -3.5, ...], ...],
         [[0.1, 0.2, -3.5, ...], ...]
       ]
     }'
   ```

**Using Python Requests:**
```python
import requests
import numpy as np

# Load your prepared data
X = np.load("data/prepared/production_X.npy")

# Single window prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"window": X[0].tolist()}
)
result = response.json()
print(f"Activity: {result['activity']}, Confidence: {result['confidence']:.2%}")

# Batch prediction (first 10 windows)
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"windows": X[:10].tolist()}
)
results = response.json()
for i, r in enumerate(results["predictions"]):
    print(f"Window {i}: {r['activity']} ({r['confidence']:.2%})")
```

**Stop the API:**
```powershell
docker-compose down inference
```

## 8) Evaluation
- If labels available, evaluate predictions:  
  ```powershell
  python src/evaluate_predictions.py `
    --predictions data/prepared/predictions/predictions.csv `
    --labels path/to/labels.csv `
    --output data/prepared/predictions/eval
  ```
- Log metrics to MLflow (pass `MLFLOW_TRACKING_URI` env).

## 9) Tracking with MLflow
- Start MLflow UI (Docker, recommended):  
  ```powershell
  docker-compose up -d mlflow
  start http://localhost:5000
  ```
- Standalone option:  
  ```powershell
  mlflow ui --backend-store-uri mlruns --default-artifact-root mlruns --host 0.0.0.0 --port 5000
  start http://localhost:5000
  ```
- In code, use `src/mlflow_tracking.py` utilities or set `MLFLOW_TRACKING_URI` before running scripts to log runs.

## 10) Docker Usage & Image Management
- Use cases in this project:
  - MLflow tracking server (`mlflow` service in `docker-compose.yml`)
  - FastAPI inference API (`inference` service)
  - Training/preprocessing on-demand (`training`, `preprocessing` profiles)

### Managing Docker Services

**Start Services:**
```powershell
# Start all default services (MLflow + Inference)
docker-compose up -d

# Start specific service only
docker-compose up -d mlflow
docker-compose up -d inference

# Start with build (if code changed)
docker-compose up -d --build inference
```

**Check Status:**
```powershell
# List running services
docker-compose ps

# View service logs
docker-compose logs -f inference
docker-compose logs -f mlflow

# Check resource usage
docker stats
```

**Stop Services:**
```powershell
# Stop all services
docker-compose down

# Stop specific service
docker-compose stop inference

# Stop and remove volumes (WARNING: deletes data!)
docker-compose down -v
```

### Managing Docker Images

**List Images:**
```powershell
# Images used by docker-compose
docker-compose images

# All local images
docker images

# Filter by name
docker images | Select-String "mlops"
```

**Remove Images:**
```powershell
# Remove dangling images (untagged)
docker image prune

# Remove specific image
docker rmi <image-id>

# Remove all unused images (CAREFUL!)
docker image prune -a

# Remove images for this project only
docker-compose down --rmi all
```

**Disk Usage:**
```powershell
# Check Docker disk usage
docker system df

# Detailed breakdown
docker system df -v
```

**Clean Everything (Nuclear Option):**
```powershell
# Stop all containers
docker-compose down

# Remove all unused containers, networks, images
docker system prune -a

# Also remove volumes (WARNING: deletes all data!)
docker system prune -a --volumes
```

## 11) Cleaning Generated Outputs & Artifacts

### Quick Clean (Safe - Only Generated Files)

**Clean preprocessing outputs:**
```powershell
# Remove processed sensor data
Remove-Item -Path "data/preprocessed/*.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/preprocessed/*.json" -Force -ErrorAction SilentlyContinue

# Remove prepared data (windowed arrays)
Remove-Item -Path "data/prepared/*.npy" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/*.json" -Force -ErrorAction SilentlyContinue

# Remove predictions
Remove-Item -Path "data/prepared/predictions/*" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "outputs/predictions/*" -Force -ErrorAction SilentlyContinue

Write-Host "✓ Cleaned preprocessing and prediction outputs"
```

**Clean logs:**
```powershell
# Remove all log files
Remove-Item -Path "logs/preprocessing/*.log" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "logs/training/*.log" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "logs/inference/*.log" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "logs/evaluation/*.log" -Force -ErrorAction SilentlyContinue

Write-Host "✓ Cleaned all log files"
```

**Clean evaluation reports:**
```powershell
# Remove evaluation outputs
Remove-Item -Path "outputs/evaluation/*.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "outputs/evaluation/*.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "outputs/analysis/*.json" -Force -ErrorAction SilentlyContinue

Write-Host "✓ Cleaned evaluation reports"
```

### Deep Clean (Be Careful!)

**Clean MLflow tracking data:**
```powershell
# WARNING: This deletes all experiment history!
# Backup first if needed: Copy-Item -Path "mlruns" -Destination "mlruns_backup" -Recurse

# Remove all MLflow runs
Remove-Item -Path "mlruns" -Recurse -Force -ErrorAction SilentlyContinue

# Remove MLflow tracking database
Remove-Item -Path "mlflow.db" -Force -ErrorAction SilentlyContinue

Write-Host "⚠️ Cleaned MLflow tracking data (all experiments deleted!)"
```

**Clean trained models (DVC-tracked):**
```powershell
# WARNING: Only do this if you can restore from DVC!
# This removes locally trained models (not pretrained)

Remove-Item -Path "models/trained/*" -Force -ErrorAction SilentlyContinue

Write-Host "⚠️ Cleaned trained models (restore with 'dvc pull' if needed)"
```

**Clean DVC cache (VERY DANGEROUS!):**
```powershell
# WARNING: This removes ALL cached data - you'll need to re-download everything!
# Only use if you're sure data is backed up to DVC remote

# Clean unused cache entries
dvc gc --workspace --cloud

# Force clean ALL cache (EXTREME!)
# Remove-Item -Path ".dvc/cache" -Recurse -Force

Write-Host "⚠️⚠️ DVC cache cleaned - run 'dvc pull' to restore data"
```

### Comprehensive Clean Script

**Create a clean script for easy reuse:**

```powershell
# Save as: scripts/clean_outputs.ps1

param(
    [switch]$All,           # Clean everything (except raw data)
    [switch]$Logs,          # Clean log files only
    [switch]$Predictions,   # Clean predictions only
    [switch]$MLflow,        # Clean MLflow tracking
    [switch]$DryRun         # Show what would be deleted (don't delete)
)

function Remove-SafeItem {
    param($Path, $Description)
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would delete: $Path ($Description)"
        return
    }
    
    $items = Get-ChildItem -Path $Path -ErrorAction SilentlyContinue
    if ($items) {
        Remove-Item -Path $Path -Force -ErrorAction SilentlyContinue
        Write-Host "✓ Deleted: $Description"
    } else {
        Write-Host "- Skipped: $Description (not found)"
    }
}

Write-Host "==================================================="
Write-Host "Pipeline Cleanup Script"
Write-Host "==================================================="
Write-Host ""

if ($All -or $Predictions) {
    Write-Host "Cleaning predictions..."
    Remove-SafeItem "data/prepared/*.npy" "Prepared data arrays"
    Remove-SafeItem "data/prepared/predictions/*" "Prediction outputs"
    Remove-SafeItem "outputs/predictions/*" "Prediction reports"
    Remove-SafeItem "data/preprocessed/*.csv" "Preprocessed CSVs"
    Write-Host ""
}

if ($All -or $Logs) {
    Write-Host "Cleaning logs..."
    Remove-SafeItem "logs/preprocessing/*.log" "Preprocessing logs"
    Remove-SafeItem "logs/training/*.log" "Training logs"
    Remove-SafeItem "logs/inference/*.log" "Inference logs"
    Remove-SafeItem "logs/evaluation/*.log" "Evaluation logs"
    Write-Host ""
}

if ($All) {
    Write-Host "Cleaning evaluation reports..."
    Remove-SafeItem "outputs/evaluation/*" "Evaluation outputs"
    Remove-SafeItem "outputs/analysis/*" "Analysis reports"
    Write-Host ""
}

if ($MLflow) {
    Write-Host "⚠️ WARNING: Cleaning MLflow tracking data!"
    $confirm = Read-Host "This will delete all experiment history. Continue? (y/N)"
    if ($confirm -eq 'y' -or $confirm -eq 'Y') {
        Remove-SafeItem "mlruns" "MLflow runs"
        Write-Host ""
    } else {
        Write-Host "- Skipped: MLflow cleanup cancelled"
        Write-Host ""
    }
}

Write-Host "==================================================="
Write-Host "Cleanup complete!"
Write-Host "==================================================="
Write-Host ""
Write-Host "To restore DVC-tracked data: dvc pull"
Write-Host "To restart MLflow tracking: docker-compose up -d mlflow"
```

**Usage examples:**
```powershell
# Dry run (see what would be deleted)
.\scripts\clean_outputs.ps1 -DryRun -All

# Clean only predictions and logs
.\scripts\clean_outputs.ps1 -Predictions -Logs

# Clean everything including MLflow
.\scripts\clean_outputs.ps1 -All -MLflow

# Clean just logs
.\scripts\clean_outputs.ps1 -Logs
```

### One-Liner Quick Clean Commands

**For daily workflow (safe):**
```powershell
# Clean before new run
Remove-Item -Path "data/prepared/*.npy","data/prepared/predictions/*","logs/*/*.log" -Force -ErrorAction SilentlyContinue; Write-Host "✓ Ready for new run"
```

**For fresh start (safe - keeps raw data and models):**
```powershell
# Complete output cleanup
Remove-Item -Path "data/preprocessed/*","data/prepared/*","outputs/*","logs/*/*.log" -Recurse -Force -ErrorAction SilentlyContinue; Write-Host "✓ All outputs cleaned"
```

**For demonstration/testing:**
```powershell
# Clean everything, restart services
docker-compose down; Remove-Item -Path "data/prepared/*","outputs/*","logs/*/*.log" -Recurse -Force -ErrorAction SilentlyContinue; docker-compose up -d; Write-Host "✓ Clean slate ready"
```

### What NOT to Delete

**KEEP THESE (unless you have backups):**
- `data/raw/*.xlsx` - Original sensor data
- `models/pretrained/*.keras` - Pre-trained models (DVC-tracked)
- `research_papers/*.csv` - Reference datasets (DVC-tracked)
- `.dvc/config` - DVC configuration
- `mlruns/` - Only delete if you don't need experiment history
- `.git/` - Never delete your Git repository!

### Recovery After Cleanup

**If you deleted too much:**
```powershell
# Restore DVC-tracked data
dvc pull

# Restore specific directory
dvc pull data/prepared.dvc
dvc pull models/pretrained.dvc

# Re-run preprocessing if needed
python src/sensor_data_pipeline.py
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --calibrate
```

## 12) Quick Resumption Checklist
- [ ] Raw Excel placed in `data/raw/` and validated
- [ ] `python src/sensor_data_pipeline.py`
- [ ] `python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --calibrate`
- [ ] DVC add + push + git commit
- [ ] (Optional) training run logged to MLflow
- [ ] Inference run saved + evaluated (batch script OR Docker API)
- [ ] MLflow UI checked for runs/artifacts (`http://localhost:5000`)
- [ ] Docker API tested (`http://localhost:8000/docs`)
- [ ] Clean outputs before next run (use cleanup commands above)
- [ ] Docker images reviewed/pruned if needed

## 13) How to Inspect Versioning, Tracking, and Docker

### DVC (Data Versioning)
```powershell
# Check what's changed
dvc status

# View experiments table
dvc exp show --html > dvc_experiments.html
start dvc_experiments.html

# Generate plots
dvc plots diff -o dvc_plots
start dvc_plots/index.html

# View dependency graph (text)
dvc dag

# View dependency graph (SVG - requires Graphviz)
dvc dag --dot | dot -Tsvg -o dvc_dag.svg
start dvc_dag.svg
```

### MLflow (Experiment Tracking)

**Start MLflow UI via Docker (Recommended):**
```powershell
docker-compose up -d mlflow
start http://localhost:5000
```

**Start MLflow UI Standalone:**
```powershell
mlflow ui --backend-store-uri mlruns --default-artifact-root mlruns --host 0.0.0.0 --port 5000
start http://localhost:5000
```

**Point Scripts to Tracking Server:**
```powershell
# For Docker MLflow server
$env:MLFLOW_TRACKING_URI="http://localhost:5000"

# For local file store
$env:MLFLOW_TRACKING_URI="file:///$(Get-Location)/mlruns"
```

**Query MLflow Programmatically:**
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# List all experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")

# Get latest run
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)
if runs:
    run = runs[0]
    print(f"Latest run: {run.info.run_id}")
    print(f"Metrics: {run.data.metrics}")
```

### Docker (Services & Images)

**Running Services:**
```powershell
# List active services with ports
docker-compose ps

# Detailed container info
docker ps -a

# Service health checks
docker inspect inference --format='{{.State.Health.Status}}'
```

**Service Logs:**
```powershell
# Follow logs for specific service
docker-compose logs -f inference

# View last 50 lines
docker-compose logs --tail=50 mlflow

# All services
docker-compose logs -f
```

**Resource Usage:**
```powershell
# Live resource monitor
docker stats

# Disk usage
docker system df

# Detailed disk breakdown
docker system df -v
```

**Inspect Images:**
```powershell
# List all images
docker images

# Filter by project
docker images | Select-String "mlops"

# Image history (layers)
docker history <image-id>

# Image details
docker inspect <image-id>
```

**Network Inspection:**
```powershell
# List Docker networks
docker network ls

# Inspect network used by services
docker network inspect masterarbeit_mlops_default

# Test connectivity between services
docker exec inference curl http://mlflow:5000/health
```

### Combined Health Check Script

```powershell
# Save as: scripts/health_check.ps1

Write-Host "==================================================="
Write-Host "MLOps Pipeline Health Check"
Write-Host "==================================================="
Write-Host ""

# Docker services
Write-Host "[Docker Services]"
$services = docker-compose ps --services
if ($services) {
    docker-compose ps
    Write-Host ""
} else {
    Write-Host "⚠️ No services running. Start with: docker-compose up -d"
    Write-Host ""
}

# MLflow
Write-Host "[MLflow]"
try {
    $mlflow = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 2
    Write-Host "✓ MLflow UI: http://localhost:5000"
} catch {
    Write-Host "✗ MLflow not accessible"
}
Write-Host ""

# Inference API
Write-Host "[Inference API]"
try {
    $api = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2
    $status = ($api.Content | ConvertFrom-Json)
    Write-Host "✓ API Health: $($status.status)"
    Write-Host "✓ Model Loaded: $($status.model_loaded)"
    Write-Host "✓ Swagger UI: http://localhost:8000/docs"
} catch {
    Write-Host "✗ Inference API not accessible"
}
Write-Host ""

# DVC status
Write-Host "[DVC Status]"
$dvcStatus = dvc status 2>&1
if ($dvcStatus -match "Data and pipelines are up to date") {
    Write-Host "✓ DVC: Up to date"
} else {
    Write-Host "⚠️ DVC: Changes detected"
    dvc status
}
Write-Host ""

# Disk usage
Write-Host "[Disk Usage]"
docker system df
Write-Host ""

# Recent files
Write-Host "[Recent Outputs]"
Write-Host "Preprocessed:"
Get-ChildItem -Path "data/preprocessed" -Filter "*.csv" -ErrorAction SilentlyContinue | 
    Select-Object -First 3 Name, LastWriteTime | Format-Table

Write-Host "Predictions:"
Get-ChildItem -Path "data/prepared/predictions" -Filter "*.csv" -ErrorAction SilentlyContinue | 
    Select-Object -First 3 Name, LastWriteTime | Format-Table

Write-Host "Logs:"
Get-ChildItem -Path "logs" -Recurse -Filter "*.log" -ErrorAction SilentlyContinue | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 5 Name, LastWriteTime | Format-Table

Write-Host "==================================================="
Write-Host "Health check complete!"
Write-Host "==================================================="
```

**Run health check:**
```powershell
.\scripts\health_check.ps1
```
