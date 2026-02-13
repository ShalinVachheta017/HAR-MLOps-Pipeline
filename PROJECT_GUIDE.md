# 📁 MasterArbeit MLOps - Complete Project Guide

> **📝 Summary:** Complete folder and file reference guide for the entire project. Contains: project structure diagram, explanation of every folder (config/, data/, src/, models/, notebooks/, docker/, docs/, etc.), file-by-file descriptions, and visual diagrams. Use this to understand what each folder/file does and where to find things.

> **Human Activity Recognition (HAR) using 1D-CNN-BiLSTM with MLOps Pipeline**  
> Master Thesis Project - Complete Folder & File Reference

---

## 🎯 Project Overview

This project implements a **production-ready MLOps pipeline** for Human Activity Recognition using smartphone sensor data (accelerometer + gyroscope). The model uses a **1D-CNN-BiLSTM architecture** to classify 11 different human activities.

### Key Technologies
```
┌─────────────────────────────────────────────────────────────┐
│  Data Versioning: DVC    │  Experiment Tracking: MLflow    │
│  Deep Learning: TensorFlow/Keras  │  Container: Docker     │
│  Pipeline: Python Scripts │  Config: YAML                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Complete Project Structure

```
MasterArbeit_MLops/
│
├── 📄 README.md                    # Main project documentation
├── 📄 PROJECT_GUIDE.md             # This file - complete reference
├── 🐳 docker-compose.yml           # Docker orchestration config
├── 🖼️ unnamed.jpg                  # Project image asset
├── 📊 dvc_experiments.html         # DVC experiments visualization
│
├── ⚙️ .gitignore                   # Git ignore rules
├── ⚙️ .dvcignore                   # DVC ignore rules
├── ⚙️ .dockerignore                # Docker ignore rules
│
├── 📂 config/                      # ⚙️ Configuration Files
├── 📂 data/                        # 📊 All Data (Raw → Processed → Prepared)
├── 📂 src/                         # 🐍 Source Code (Pipeline Scripts)
├── 📂 models/                      # 🧠 ML Models (Pretrained & Trained)
├── 📂 notebooks/                   # 📓 Jupyter Notebooks
├── 📂 docker/                      # 🐳 Docker Configuration
├── 📂 docs/                        # 📚 Documentation
├── 📂 logs/                        # 📋 Execution Logs
├── 📂 outputs/                     # 📈 Pipeline Outputs
├── 📂 mlruns/                      # 🔬 MLflow Experiments
├── 📂 tests/                       # 🧪 Unit Tests
├── 📂 research_papers/             # 📄 Reference Papers
├── 📂 images/                      # 🖼️ Project Images
├── 📂 cheat sheet/                 # 📝 Quick Reference Guides
├── 📂 .dvc/                        # DVC Internal Files
└── 📂 .dvc_storage/                # DVC Local Cache
```

---

## 📂 Detailed Folder Breakdown

### 📂 `config/` - Configuration Files
```
config/
├── 📄 pipeline_config.yaml     # Main pipeline configuration
│                               # - Data paths, model settings
│                               # - Preprocessing parameters
│                               # - Training hyperparameters
│
├── 📄 mlflow_config.yaml       # MLflow tracking configuration
│                               # - Experiment names
│                               # - Tracking URI settings
│
├── 📄 requirements.txt         # Python dependencies
│                               # - All pip packages needed
│
└── 📄 .pylintrc                # Python linting rules
                                # - Code quality settings
```

**🎯 Purpose:** Centralized configuration for reproducible experiments

---

### 📂 `data/` - Data Storage
```
data/
├── 📂 raw/                         # 🔴 Original unprocessed data
│   ├── 📊 accelerometer_data.xlsx  # Raw accelerometer readings
│   ├── 📊 gyroscope_data.xlsx      # Raw gyroscope readings
│   └── 📊 all_users_data_labeled.csv # Training data (2005 dataset)
│
├── 📂 preprocessed/                # 🟡 After sensor fusion
│   ├── 📊 sensor_fused_50Hz.csv    # Resampled to 50Hz
│   ├── 📊 sensor_merged_native_rate.csv # Native rate merged
│   └── 📄 sensor_fused_meta.json   # Preprocessing metadata
│
├── 📂 processed/                   # 🟢 DVC tracked processed data
│   └── (DVC managed files)
│
├── 📂 prepared/                    # ✅ Ready for inference
│   ├── 📊 production_X.npy         # Windowed data arrays
│   ├── 📄 production_metadata.json # Data metadata
│   ├── 📄 config.json              # Preparation config
│   ├── 📂 predictions/             # Model predictions
│   │   ├── predictions_*.csv       # Predicted labels
│   │   ├── predictions_*_probs.npy # Probability scores
│   │   └── predictions_*_metadata.json
│   └── 📄 *.md                     # Data documentation
│
├── 📂 samples_2005 dataset/        # Sample reference data
│
├── 📄 raw.dvc                      # DVC tracking file
├── 📄 processed.dvc                # DVC tracking file
└── 📄 prepared.dvc                 # DVC tracking file
```

**🎯 Purpose:** Data versioning with DVC, clear data lineage from raw → prepared

**📊 Data Flow:**
```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
│   RAW    │ → │ PREPROCESSED │ → │ PROCESSED │ → │ PREPARED │
│ .xlsx    │    │ sensor_fused │    │ (DVC)     │    │ .npy     │
└──────────┘    └──────────────┘    └───────────┘    └──────────┘
     ↓                ↓                   ↓               ↓
  Original      50Hz Resample      Versioned       Inference-Ready
```

---

### 📂 `src/` - Source Code
```
src/
├── 🐍 run_inference.py         # 🚀 MAIN: Model inference pipeline
│                               # - Loads model & data
│                               # - Runs predictions
│                               # - MLflow tracking integration
│
├── 🐍 preprocess_data.py       # Data preprocessing script
│                               # - Sensor fusion (accel + gyro)
│                               # - Resampling to 50Hz
│                               # - Domain calibration
│
├── 🐍 sensor_data_pipeline.py  # Core sensor processing
│                               # - Data loading utilities
│                               # - Windowing functions
│                               # - Feature extraction
│
├── 🐍 evaluate_predictions.py  # Prediction evaluation
│                               # - Confusion matrix
│                               # - Per-class metrics
│                               # - Activity distribution
│
├── 🐍 data_validator.py        # Data validation checks
│                               # - Schema validation
│                               # - Range checks
│                               # - Missing value detection
│
├── 🐍 compare_data.py          # Data comparison utilities
│                               # - Compare datasets
│                               # - Distribution analysis
│
├── 🐍 mlflow_tracking.py       # MLflow utilities
│                               # - Experiment setup
│                               # - Metric logging helpers
│
├── 🐍 config.py                # Configuration loader
│                               # - Load YAML configs
│                               # - Path management
│
├── 📄 README.md                # Source code documentation
│
└── 📂 Archived(...)/           # 📦 Old/unused scripts
    ├── prepare_production_data.py
    ├── prepare_training_data.py
    └── convert_production_units.py
```

**🎯 Purpose:** All executable Python code for the MLOps pipeline

**🔄 Pipeline Execution Order:**
```
1. preprocess_data.py    → Sensor fusion & resampling
2. run_inference.py      → Model predictions (with MLflow)
3. evaluate_predictions.py → Metrics & analysis
```

---

### 📂 `models/` - Machine Learning Models
```
models/
├── 📂 pretrained/                              # Pre-trained models
│   ├── 🧠 fine_tuned_model_1dcnnbilstm.keras  # Main HAR model
│   │                                           # - 499,131 parameters
│   │                                           # - Input: (200, 6)
│   │                                           # - Output: 11 classes
│   │
│   └── 📄 model_info.json                     # Model metadata
│                                               # - Architecture details
│                                               # - Training info
│
├── 📂 trained/                                # Models trained in this project
│   └── (New models go here)
│
└── 📄 pretrained.dvc                          # DVC tracking
```

**🎯 Purpose:** Model versioning and storage

**🧠 Model Architecture:**
```
┌─────────────────────────────────────────────────┐
│         1D-CNN-BiLSTM Architecture              │
├─────────────────────────────────────────────────┤
│  Input: (batch, 200 timesteps, 6 features)      │
│    ↓                                            │
│  Conv1D Layers → Feature extraction             │
│    ↓                                            │
│  Bidirectional LSTM → Temporal patterns         │
│    ↓                                            │
│  Dense Layers → Classification                  │
│    ↓                                            │
│  Output: 11 activity classes                    │
└─────────────────────────────────────────────────┘
```

**📋 11 Activity Classes:**
```
0: sit               5: stairsup        10: forehead_rubbing
1: stand             6: stairsdown
2: walk              7: run
3: bike              8: car
4: e-bike            9: bus
```

---

### 📂 `notebooks/` - Jupyter Notebooks
```
notebooks/
├── 📓 data_preprocessing_step1.ipynb   # Step-by-step preprocessing
│                                        # - Interactive data exploration
│                                        # - Sensor fusion walkthrough
│
├── 📓 production_preprocessing.ipynb   # Production data prep
│                                        # - Production pipeline demo
│
├── 📓 data_comparison.ipynb            # Data analysis
│                                        # - Compare datasets
│                                        # - Distribution plots
│
├── 📓 from_guide_processing.ipynb      # Guide-based processing
│
├── 📓 scalable.ipynb                   # Scalability experiments
│
├── 📂 exploration/                     # Experimental notebooks
│   └── (Draft/experimental work)
│
└── 📄 README.md                        # Notebook descriptions
```

**🎯 Purpose:** Interactive development, exploration, and documentation

---

### 📂 `docker/` - Containerization
```
docker/
├── 🐳 Dockerfile.inference     # Inference container
│                               # - Lightweight for predictions
│                               # - TensorFlow runtime
│
├── 🐳 Dockerfile.training      # Training container
│                               # - Full training environment
│                               # - GPU support ready
│
└── 📂 api/                     # FastAPI application
    ├── 🐍 main.py              # API endpoints
    │                           # - /predict endpoint
    │                           # - /health endpoint
    │
    └── 🐍 __init__.py          # Package init
```

**🎯 Purpose:** Reproducible containerized deployment

**🐳 Docker Usage:**
```bash
# Build inference container
docker build -f docker/Dockerfile.inference -t har-inference .

# Run with docker-compose
docker-compose up
```

---

### 📂 `outputs/` - Pipeline Outputs
```
outputs/
├── 📂 evaluation/                      # Evaluation results
│   ├── 📄 evaluation_*.json            # Metrics in JSON
│   └── 📄 evaluation_*.txt             # Human-readable report
│
└── 🖼️ gravity_removal_comparison.png  # Visualization output
```

**🎯 Purpose:** Store all pipeline outputs (evaluations, visualizations)

---

### 📂 `logs/` - Execution Logs
```
logs/
├── 📂 preprocessing/     # Preprocessing logs
├── 📂 inference/         # Inference logs
├── 📂 training/          # Training logs
└── 📂 evaluation/        # Evaluation logs
```

**🎯 Purpose:** Debugging and audit trail

---

### 📂 `mlruns/` - MLflow Experiment Tracking
```
mlruns/
├── 📂 0/                           # Default experiment
├── 📂 950614147457743858/          # HAR experiments
│   └── 📂 <run_id>/                # Individual runs
│       ├── 📂 metrics/             # Logged metrics
│       ├── 📂 params/              # Hyperparameters
│       └── 📂 artifacts/           # Saved outputs
│
├── 📂 models/                      # Registered models
└── 📂 .trash/                      # Deleted runs
```

**🎯 Purpose:** Track all experiments, compare runs, reproduce results

**🔬 View MLflow UI:**
```bash
mlflow ui --backend-store-uri mlruns
# Open: http://localhost:5000
```

---

### 📂 `docs/` - Documentation
```
docs/
├── 📄 CONCEPTS_EXPLAINED.md        # Technical concepts
├── 📄 CURRENT_STATUS.md            # Project status
├── 📄 FILE_ORGANIZATION_SUMMARY.md # File organization
├── 📄 FRESH_START_CLEANUP_GUIDE.md # Cleanup instructions
├── 📄 MARKDOWN_CLEANUP_GUIDE.md    # Doc organization
├── 📄 PIPELINE_RERUN_GUIDE.md      # Pipeline execution
├── 📄 RESEARCH_PAPERS_ANALYSIS.md  # Paper summaries
├── 📄 SRC_FOLDER_ANALYSIS.md       # Source code analysis
│
└── 📂 archived/                    # 📦 Archived documentation
    ├── 🗑️ DELETE_*.md              # Can be deleted
    └── 📌 KEEP_*.md                # Keep for reference
```

**🎯 Purpose:** All project documentation organized in one place

---

### 📂 `research_papers/` - Reference Materials
```
research_papers/
├── 📄 1806.05208v2.pdf             # HAR research paper
├── 📄 2202.10169v2.pdf             # Deep learning paper
├── 📄 EHB_2025_71.pdf              # Behavior analysis
├── 📄 ICTH_16.pdf                  # Telehealth paper
├── 📊 Final_resorecs_paper_list.xlsx # Paper list
├── 📊 anxiety_dataset.csv          # Reference dataset
└── 📓 temp.ipynb                   # Paper analysis notebook
```

**🎯 Purpose:** Research papers and reference materials for thesis

---

### 📂 `tests/` - Unit Tests
```
tests/
└── (Empty - tests to be added)
```

**🎯 Purpose:** Automated testing (to be implemented)

---

## 🔄 Pipeline Workflow Visual

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline Flow                               │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
    │  📊 RAW DATA │         │  ⚙️ CONFIG   │         │  🧠 MODEL    │
    │  data/raw/   │         │  config/     │         │  models/     │
    └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
           │                        │                        │
           ▼                        ▼                        │
    ┌──────────────────────────────────────────────┐        │
    │        1️⃣ PREPROCESSING                      │        │
    │        src/preprocess_data.py                │        │
    │        ─────────────────────                 │        │
    │        • Sensor fusion (accel + gyro)        │        │
    │        • Resample to 50Hz                    │        │
    │        • Domain calibration (-6.295 Az)      │        │
    └──────────────┬───────────────────────────────┘        │
                   │                                        │
                   ▼                                        │
    ┌──────────────────────────────────────────────┐        │
    │        📂 data/preprocessed/                 │        │
    │        sensor_fused_50Hz.csv                 │        │
    └──────────────┬───────────────────────────────┘        │
                   │                                        │
                   ▼                                        ▼
    ┌──────────────────────────────────────────────────────────┐
    │        2️⃣ INFERENCE                                      │
    │        src/run_inference.py                              │
    │        ─────────────────────                             │
    │        • Window data (200 samples, 50% overlap)          │
    │        • Load 1D-CNN-BiLSTM model                        │
    │        • Run predictions                                 │
    │        • MLflow tracking                                 │
    └──────────────┬───────────────────────────────────────────┘
                   │
                   ├──────────────────────────────────────┐
                   ▼                                      ▼
    ┌──────────────────────────────┐    ┌──────────────────────────────┐
    │  📂 data/prepared/predictions│    │  📂 mlruns/                  │
    │  predictions_*.csv           │    │  Experiment tracking         │
    │  predictions_*_probs.npy     │    │  Metrics, params, artifacts  │
    └──────────────┬───────────────┘    └──────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────┐
    │        3️⃣ EVALUATION                         │
    │        src/evaluate_predictions.py           │
    │        ─────────────────────                 │
    │        • Confusion matrix                    │
    │        • Per-class precision/recall          │
    │        • Activity distribution               │
    └──────────────┬───────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────┐
    │        📂 outputs/evaluation/                │
    │        evaluation_*.json                     │
    │        evaluation_*.txt                      │
    └──────────────────────────────────────────────┘
```

---

## 📋 Quick Reference Commands

### Data Operations (DVC)
```bash
# Pull data from remote
dvc pull

# Push data to remote
dvc push

# Check data status
dvc status

# Reproduce pipeline
dvc repro
```

### MLflow Operations
```bash
# Start MLflow UI
mlflow ui --backend-store-uri mlruns

# View at http://localhost:5000
```

### Pipeline Execution
```bash
# Run preprocessing
python src/preprocess_data.py

# Run inference
python src/run_inference.py

# Run evaluation
python src/evaluate_predictions.py
```

### Docker Operations
```bash
# Build and run
docker-compose up --build

# Inference only
docker build -f docker/Dockerfile.inference -t har-inference .
docker run -v $(pwd)/data:/app/data har-inference
```

---

## 📊 Key Project Statistics

| Metric | Value |
|--------|-------|
| **Model Parameters** | 499,131 |
| **Input Shape** | (200 timesteps, 6 features) |
| **Output Classes** | 11 activities |
| **Sampling Rate** | 50Hz |
| **Window Size** | 200 samples (4 seconds) |
| **Window Overlap** | 50% |
| **Domain Calibration** | -6.295 m/s² (Az offset) |

---

## 📚 Documentation Index

| File | Location | Purpose |
|------|----------|---------|
| [README.md](README.md) | Root | Main project documentation |
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | Root | This complete reference |
| [CONCEPTS_EXPLAINED.md](docs/CONCEPTS_EXPLAINED.md) | docs/ | Technical concepts |
| [PIPELINE_RERUN_GUIDE.md](docs/PIPELINE_RERUN_GUIDE.md) | docs/ | How to run pipeline |
| [SRC_FOLDER_ANALYSIS.md](docs/SRC_FOLDER_ANALYSIS.md) | docs/ | Source code details |

---

## 🗑️ Archived Documentation

Files in `docs/archived/` are organized by usefulness:

| Prefix | Meaning | Action |
|--------|---------|--------|
| `DELETE_*` | Outdated/redundant | Safe to delete |
| `KEEP_*` | Useful for future | Keep for reference |
| (no prefix) | Review needed | Check before deleting |

---

*Last Updated: December 2024*
