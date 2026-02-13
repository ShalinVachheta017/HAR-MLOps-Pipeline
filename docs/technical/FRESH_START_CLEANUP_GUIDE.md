# Complete Fresh Start Guide - Clean All Old Records & DVC Cache

> **📝 Summary:** Commands to delete all generated outputs and start fresh. Contains: what gets deleted (evaluation reports, MLflow data, logs, predictions, DVC cache), PowerShell commands, and safety notes. Use this when you want to reset the pipeline to a clean state.

**Date:** December 12, 2025  
**Purpose:** Delete all old generated files and start the pipeline from scratch (zero)  
**Safety Level:** SAFE - Only removes outputs, logs, and old DVC cache

---

## 🎯 What We're Deleting (Complete Clean Slate)

### Files & Folders to DELETE:
1. **Old Evaluation Reports** - 8 files from past runs
2. **MLflow Tracking Data** - All experiment history
3. **Old Logs** - All old pipeline execution logs  
4. **Old Predictions** - All past prediction results
5. **Old Preprocessed Data** - Sensor fusion CSVs from old runs
6. **DVC Cache** - Old cached versions (will re-download if needed)

### Files & Folders to KEEP:
1. **Raw Data** - `data/raw/*.xlsx` (original Garmin exports)
2. **Pretrained Model** - `models/pretrained/*.keras` (fine-tuned model)
3. **Reference Datasets** - `research_papers/*.csv` (anxiety & Garmin labeled)
4. **Code & Configs** - All source code, configs, Docker files
5. **Git Repository** - `.git/` folder (version control)

---

## 🧹 Step-by-Step Cleanup

### Method 1: PowerShell (Recommended - Fast & Safe)

**Run this PowerShell command:**

```powershell
# ============================================================
# COMPREHENSIVE CLEANUP - DELETE ALL OLD GENERATED FILES
# ============================================================

Write-Host "🧹 Cleaning old pipeline outputs..." -ForegroundColor Yellow
Write-Host ""

# 1. Remove old evaluation reports (8 files)
Write-Host "[1/6] Removing evaluation reports..."
Remove-Item -Path "outputs/evaluation/*.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "outputs/evaluation/*.txt" -Force -ErrorAction SilentlyContinue
Write-Host "✓ Deleted evaluation reports"

# 2. Remove old logs (all log files)
Write-Host "[2/6] Removing pipeline logs..."
Remove-Item -Path "logs/preprocessing/*.log" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "logs/training/*.log" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "logs/inference/*.log" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "logs/evaluation/*.log" -Force -ErrorAction SilentlyContinue
Write-Host "✓ Deleted all logs"

# 3. Remove old predictions
Write-Host "[3/6] Removing prediction outputs..."
Remove-Item -Path "data/prepared/predictions/*.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/predictions/*.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/predictions/*.npy" -Force -ErrorAction SilentlyContinue
Write-Host "✓ Deleted predictions"

# 4. Remove old preprocessed data (sensor fusion CSVs)
Write-Host "[4/6] Removing old preprocessed data..."
Remove-Item -Path "data/preprocessed/*.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/preprocessed/*.json" -Force -ErrorAction SilentlyContinue
Write-Host "✓ Deleted preprocessed data"

# 5. Remove prepared data (windowed arrays)
Write-Host "[5/6] Removing prepared data (NPY arrays)..."
Remove-Item -Path "data/prepared/*.npy" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/config.json" -Force -ErrorAction SilentlyContinue
Write-Host "✓ Deleted prepared data arrays"

# 6. Remove MLflow tracking data (WARNING: deletes all experiments!)
Write-Host "[6/6] Removing MLflow tracking data..." -ForegroundColor Cyan
Write-Host "⚠️ This will delete all past experiments from MLflow"
$confirm = Read-Host "Continue? Type 'YES' to confirm"
if ($confirm -eq "YES") {
    Remove-Item -Path "mlruns" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "mlflow.db" -Force -ErrorAction SilentlyContinue
    Write-Host "✓ Deleted MLflow data"
} else {
    Write-Host "- Skipped MLflow cleanup"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "✅ CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your workspace is now ready to restart from zero."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Verify raw data: ls data/raw/"
Write-Host "  2. Verify pretrained model: ls models/pretrained/"
Write-Host "  3. Start fresh run: python src/sensor_data_pipeline.py"
Write-Host ""
```

Save this as a script and run:
```powershell
# Option A: Run directly
& {
    # ... paste the code above ...
}

# Option B: Save as file and run
# Save to: scripts/fresh_start.ps1
.\scripts\fresh_start.ps1
```

---

### Method 2: Manual Deletion (If you prefer file explorer)

**Delete these folders/files one by one:**

#### A. Evaluation Reports (SAFE to delete)
```
outputs/evaluation/
├── evaluation_20251208_145052.json     ❌ DELETE
├── evaluation_20251208_145052.txt      ❌ DELETE
├── evaluation_20251211_222024.json     ❌ DELETE
├── evaluation_20251211_222024.txt      ❌ DELETE
├── evaluation_20251211_222741.json     ❌ DELETE
├── evaluation_20251211_222741.txt      ❌ DELETE
├── evaluation_20251211_225323.json     ❌ DELETE
└── evaluation_20251211_225323.txt      ❌ DELETE
```

#### B. Logs (SAFE to delete)
```
logs/
├── preprocessing/*.log                  ❌ DELETE ALL
├── training/*.log                       ❌ DELETE ALL
├── inference/*.log                      ❌ DELETE ALL
└── evaluation/*.log                     ❌ DELETE ALL
```

#### C. Predictions (SAFE to delete)
```
data/prepared/predictions/
├── *.csv                                ❌ DELETE ALL
├── *.json                               ❌ DELETE ALL
└── *.npy                                ❌ DELETE ALL
```

#### D. Preprocessed Data (SAFE to delete)
```
data/preprocessed/
├── sensor_fused_50Hz.csv               ❌ DELETE
├── sensor_merged_native_rate.csv       ❌ DELETE
└── sensor_fused_meta.json              ❌ DELETE
```

#### E. Prepared Data Arrays (SAFE to delete)
```
data/prepared/
├── production_X.npy                     ❌ DELETE
├── production_metadata.json             ❌ DELETE
├── config.json                          ❌ DELETE (will be regenerated)
└── predictions/                         ❌ DELETE (already empty)
```

#### F. MLflow Data (CAREFUL - deletes experiment history)
```
mlruns/                                  ⚠️ DELETE (if you don't need history)
mlflow.db                                ⚠️ DELETE
```

---

## 📊 DVC Cache Management

### Understanding DVC Files

Your DVC tracking files (.dvc) are stored in Git and point to actual data in cloud storage (or local cache).

**Current DVC Files (KEEP THESE):**
```
data/raw.dvc                    ✅ KEEP - Points to raw sensor data
data/processed.dvc              ✅ KEEP - Points to processed/fused data
data/prepared.dvc               ✅ KEEP - Points to windowed arrays
models/pretrained.dvc           ✅ KEEP - Points to pre-trained model
research_papers/*.csv.dvc       ✅ KEEP - Points to reference datasets
```

These `.dvc` files are small (kilobytes) and tracked in Git. They're pointers to the actual large files in DVC storage.

### Clean DVC Cache (Optional)

```powershell
# See what's in DVC cache
dvc cache dir

# Show cache usage
dvc cache status

# Clean unused cache entries (safe)
dvc gc --workspace

# Clean ALL cache (EXTREME - you'll need to re-download everything)
# Remove-Item -Path ".dvc/cache" -Recurse -Force
```

### Re-download Data After Cleanup

If you clean everything, restore from DVC:

```powershell
# Pull all tracked data back
dvc pull

# Or pull specific directories
dvc pull data/raw.dvc
dvc pull models/pretrained.dvc
```

---

## 🔄 Starting Fresh - Step by Step

### After Cleanup, Follow This Sequence:

**Step 1: Verify Core Files Still Exist**
```powershell
# Check raw data (should have 2 Excel files)
ls data/raw/*.xlsx

# Check pretrained model (should be ~18MB)
ls models/pretrained/*.keras

# Check Git is intact
git log --oneline | head -5
```

**Step 2: Clear and Prepare**
```powershell
# Make sure DVC is up to date
dvc pull

# Verify Python environment
conda activate thesis-mlops
pip list | grep -E "tensorflow|scikit|pandas"
```

**Step 3: Fresh Pipeline Run**
```powershell
# Step 1: Sensor fusion (raw → fused CSV)
python src/sensor_data_pipeline.py

# Step 2: Preprocessing (CSV → windowed arrays)
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --calibrate

# Step 3: Inference (arrays → predictions)
python src/run_inference.py

# Step 4: Evaluation (predictions → reports)
python src/evaluate_predictions.py
```

**Step 4: Start Services**
```powershell
# MLflow for tracking
docker-compose up -d mlflow
start http://localhost:5000

# Inference API
docker-compose up -d inference
start http://localhost:8000/docs
```

---

## 📝 Updated data/prepared Folder Structure (After Fresh Start)

**Currently has:**
```
data/prepared/
├── config.json                  ← Scaler config (OK to keep)
├── PRODUCTION_DATA_README.md    ← Documentation (OK to keep)
├── production_metadata.json     ← Metadata (can delete, will regenerate)
└── predictions/                 ← Empty folder for new predictions
```

**After fresh run will have:**
```
data/prepared/
├── production_X.npy             ← New prepared arrays
├── production_metadata.json     ← New metadata
├── config.json                  ← Scaler config
├── predictions/
│   ├── predictions_*.csv        ← New predictions
│   └── predictions_*.json       ← New metadata
└── PRODUCTION_DATA_README.md    ← Documentation
```

---

## 🆘 Recovery (If Cleanup Goes Wrong)

### If you accidentally deleted something important:

```powershell
# Check Git for deleted files
git log --full-history -- <filepath>

# Restore from Git
git restore <filepath>

# Restore from DVC
dvc pull <file.dvc>

# Restore everything
git restore .
dvc checkout
```

### If you lost MLflow data:

```powershell
# MLflow data can't be recovered (it's not in Git or DVC)
# But you can restart it fresh:
docker-compose up -d mlflow
# It will create new empty mlruns/ folder
```

---

## 📋 Verification Checklist After Cleanup

Run this to verify your fresh start:

```powershell
# ============================================================
# POST-CLEANUP VERIFICATION CHECKLIST
# ============================================================

Write-Host "Verifying clean slate..." -ForegroundColor Green
Write-Host ""

# 1. Check raw data exists
$rawCount = (Get-ChildItem data/raw -Filter *.xlsx -ErrorAction SilentlyContinue).Count
Write-Host "[✓] Raw data files: $rawCount (should be 2)"

# 2. Check model exists
$modelSize = (Get-Item models/pretrained/*.keras -ErrorAction SilentlyContinue).Length
Write-Host "[✓] Model size: $([math]::Round($modelSize/1MB, 1)) MB (should be ~18MB)"

# 3. Check no old predictions
$oldPreds = (Get-ChildItem data/prepared/predictions -Filter *.csv -ErrorAction SilentlyContinue).Count
Write-Host "[?] Old predictions: $oldPreds (should be 0)"

# 4. Check no old logs
$oldLogs = (Get-ChildItem logs -Recurse -Filter *.log -ErrorAction SilentlyContinue).Count
Write-Host "[?] Old logs: $oldLogs (should be 0)"

# 5. Check no old evaluations
$oldEvals = (Get-ChildItem outputs/evaluation -Filter *.json -ErrorAction SilentlyContinue).Count
Write-Host "[?] Old evaluations: $oldEvals (should be 0)"

# 6. Check Git is clean
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "[!] Git has changes: $gitStatus"
} else {
    Write-Host "[✓] Git is clean (no uncommitted changes)"
}

# 7. Check DVC is in sync
dvc status

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Verification complete! Ready for fresh start." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
```

---

## 🚀 Complete Fresh Start PowerShell Script

Save this complete script:

```powershell
# File: scripts/complete_fresh_start.ps1
# Purpose: One-command complete cleanup + fresh start

param(
    [switch]$SkipMLflow,  # Don't delete MLflow data
    [switch]$DryRun       # Show what would be deleted
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🧹 COMPLETE FRESH START CLEANUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "⚠️  DRY RUN MODE - No files will actually be deleted" -ForegroundColor Yellow
    Write-Host ""
}

# Function to safely delete
function Remove-SafeItem {
    param($Path, $Description)
    
    $items = Get-ChildItem -Path $Path -ErrorAction SilentlyContinue
    if ($items) {
        if ($DryRun) {
            Write-Host "  [Would delete] $Description"
        } else {
            Remove-Item -Path $Path -Force -ErrorAction SilentlyContinue
            Write-Host "  ✓ $Description"
        }
    }
}

# Cleanup tasks
$tasks = @(
    @{Name="Evaluation reports"; Path="outputs/evaluation/*"},
    @{Name="Preprocessing logs"; Path="logs/preprocessing/*"},
    @{Name="Training logs"; Path="logs/training/*"},
    @{Name="Inference logs"; Path="logs/inference/*"},
    @{Name="Evaluation logs"; Path="logs/evaluation/*"},
    @{Name="Old predictions"; Path="data/prepared/predictions/*"},
    @{Name="Old preprocessed data"; Path="data/preprocessed/*"},
    @{Name="Prepared arrays"; Path="data/prepared/*.npy"}
)

foreach ($task in $tasks) {
    Write-Host "Removing $($task.Name)..."
    Get-ChildItem -Path $task.Path -ErrorAction SilentlyContinue -Recurse -File | 
        Where-Object { -not $DryRun } | Remove-Item -Force -ErrorAction SilentlyContinue
    if ($DryRun) {
        Write-Host "  [Would delete files matching: $($task.Path)]"
    } else {
        Write-Host "  ✓ Done"
    }
}

# MLflow cleanup (if not skipped)
if (-not $SkipMLflow) {
    Write-Host "Removing MLflow tracking data..."
    if ($DryRun) {
        Write-Host "  [Would delete] mlruns/ folder"
        Write-Host "  [Would delete] mlflow.db"
    } else {
        Remove-Item -Path "mlruns" -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Path "mlflow.db" -Force -ErrorAction SilentlyContinue
        Write-Host "  ✓ Done"
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
if ($DryRun) {
    Write-Host "✓ DRY RUN COMPLETE - Use -DryRun:$false to execute" -ForegroundColor Yellow
} else {
    Write-Host "✅ CLEANUP COMPLETE!" -ForegroundColor Green
}
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. python src/sensor_data_pipeline.py"
Write-Host "  2. python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --calibrate"
Write-Host "  3. python src/run_inference.py"
Write-Host "  4. python src/evaluate_predictions.py"
Write-Host ""
```

**Usage:**
```powershell
# Dry run (see what would be deleted)
.\scripts\complete_fresh_start.ps1 -DryRun

# Actually delete
.\scripts\complete_fresh_start.ps1

# Don't delete MLflow data
.\scripts\complete_fresh_start.ps1 -SkipMLflow
```

---

## 📌 Important Notes

1. **DVC files (.dvc) stay in Git** - They're small pointers, not actual data
2. **Raw data is safe** - Stored in DVC remote, easily restored with `dvc pull`
3. **Pretrained model is safe** - DVC-tracked, can restore anytime
4. **MLflow data is NOT versioned** - Once deleted, experiments are gone forever
5. **All logs are regenerated** - New logs created on each pipeline run

---

**Ready to start fresh? Run the PowerShell script above! 🚀**
