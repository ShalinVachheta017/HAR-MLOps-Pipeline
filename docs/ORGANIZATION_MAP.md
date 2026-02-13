# Repository Organization Map

**Date:** January 15, 2026  
**Purpose:** Guide to organized documentation and papers structure

---

## 📁 Folder Structure

```
MasterArbeit_MLops/
├── docs/
│   ├── thesis/          → Thesis-critical content
│   ├── research/        → Literature analysis
│   ├── technical/       → Implementation details
│   └── archive/         → Reference/old files
├── papers/
│   ├── anxiety_detection/      → HAR & mental health papers
│   ├── domain_adaptation/      → Transfer learning papers
│   ├── mlops_production/       → MLOps & deployment papers
│   └── uncertainty_confidence/ → Calibration & OOD papers
└── [Root files]         → Active planning documents
```

---

## 📚 docs/thesis/ — Thesis Writing Materials

### Core Chapters

| File | Thesis Chapter | Content |
|------|----------------|---------|
| **UNLABELED_EVALUATION.md** | Chapter 3: Methodology | 4-layer monitoring framework |
| **THESIS_READY_UNLABELED_EVALUATION_PLAN.md** | Chapter 4: Evaluation | Ready-to-use evaluation text |
| **CONCEPTS_EXPLAINED.md** | Chapter 2: Background | Key ML/HAR concepts |
| **HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md** | Chapter 5: Discussion | Domain shift analysis & limitations |
| **QA_LAB_TO_LIFE_GAP.md** | Chapter 5: Discussion | Lab-to-production gap |

### Supporting Material

| File | Use |
|------|-----|
| **KEEP_Technology_Stack_Analysis.md** | Chapter 3: Implementation justification |
| **KEEP_Production_Robustness_Guide.md** | Chapter 6: Future work |
| **KEEP_Reference_Project_Learnings.md** | Related work & best practices |
| **FINE_TUNING_STRATEGY.md** | Model adaptation methodology |
| **FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md** | Timeline & progress tracking |

---

## 🔬 docs/research/ — Literature Review

| File | Content |
|------|---------|
| **RESEARCH_PAPERS_ANALYSIS.md** | ICTH_16 & EHB_2025_71 deep analysis |
| **RESEARCH_PAPER_INSIGHTS.md** | Cross-paper synthesis |
| **KEEP_Research_QA_From_Papers.md** | Q&A from literature (UDA, retraining, MLOps) |

**Use for:** Chapter 2 (Related Work), justifying design decisions

---

## 🔧 docs/technical/ — Implementation Details

| File | Purpose |
|------|---------|
| **PIPELINE_VISUALIZATION_CURRENT.md** | System architecture diagrams |
| **PIPELINE_TEST_RESULTS.md** | Execution results & validation |
| **PIPELINE_RERUN_GUIDE.md** | Reproduction instructions |
| **pipeline_audit_map.md** | Code inventory |
| **evaluation_audit.md** | Evaluation framework audit |
| **tracking_audit.md** | MLflow/DVC setup |
| **QC_EXECUTION_SUMMARY.md** | Quality control results |
| **root_cause_low_accuracy.md** | Debugging analysis |

**Use for:** Chapter 3 (Implementation), Appendix (technical details)

---

## 📦 docs/archive/ — Reference Files

| File | Note |
|------|------|
| **FRESH_START_CLEANUP_GUIDE.md** | Historical cleanup documentation |
| **RESTRUCTURE_PIPELINE_PACKAGES.md** | Old refactoring plans |
| **Mondaymeet.md** | Meeting notes |
| **LATER_Offline_MLOps_Guide.md** | Future offline deployment ideas |
| **extranotes.md** | Miscellaneous notes |

**Use for:** Context/history, but not primary thesis content

---

## 📄 Root-Level Files (Keep at Root)

| File | Why at Root |
|------|-------------|
| **README.md** | Repository overview |
| **Thesis_Plan.md** | Master thesis structure |
| **FINAL_3_PATHWAYS_TO_COMPLETE_THESIS.md** | Active completion roadmap |
| **PROJECT_GUIDE.md** | Quick start guide |
| **MASTER_FILE_ANALYSIS_AND_NEXT_STEPS.md** | Current status tracker |
| **ans.md** | Quick notes/scratch |

---

## 📚 papers/ — Research Papers by Topic

### papers/anxiety_detection/

Papers on mental health monitoring, anxiety detection, wearable sensors:

- **ADAM-sense_Anxietydisplayingactivitiesrecognitionby.pdf** — Core dataset paper
- **A Survey on Wearable Sensors for Mental Health Monitoring.pdf**
- **Anxiety Detection Leveraging Mobile Passive Sensing.pdf**
- **Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables.pdf**
- All "anxiety", "mental health", "wearable sensors" papers

### papers/domain_adaptation/

Papers on transfer learning, cross-position HAR, sensor placement:

- **Domain Adaptation for Inertial Measurement Unit-based Human.pdf**
- **Transfer Learning in Human Activity Recognition  A Survey.pdf**
- Papers with "domain adaptation", "transfer learning", "cross-position"

### papers/mlops_production/

Papers on MLOps, deployment, monitoring, CI/CD:

- **Building-Scalable-MLOps-Optimizing-Machine-Learning-Deployment-and-Operations.pdf**
- **Practical-mlops-operationalizing-machine-learning-models.pdf**
- **Demystifying MLOps and Presenting a Recipe for the Selection of Open-Source Tools 2021.pdf**
- **Machine Learning Operations in Health Care A Scoping Review.pdf**
- All papers with "MLOps", "deployment", "production"

### papers/uncertainty_confidence/

Papers on calibration, OOD detection, confidence estimation:

- **When Does Optimizing a Proper Loss Yield Calibration.pdf**
- **NeurIPS-2020-energy-based-out-of-distribution-detection-Paper.pdf**
- **NeurIPS-2021-adaptive-conformal-inference-under-distribution-shift-Paper.pdf**
- Papers on "calibration", "uncertainty", "OOD detection"

---

## 🎯 Quick Navigation for Thesis Writing

### Chapter 1: Introduction
- Use: **Thesis_Plan.md**, **PROJECT_GUIDE.md**

### Chapter 2: Background & Related Work
- Use: **docs/thesis/CONCEPTS_EXPLAINED.md**
- Use: **docs/research/RESEARCH_PAPERS_ANALYSIS.md**
- Use: **docs/research/KEEP_Research_QA_From_Papers.md**
- Cite: **papers/anxiety_detection/**, **papers/mlops_production/**

### Chapter 3: Methodology
- Use: **docs/thesis/UNLABELED_EVALUATION.md**
- Use: **docs/thesis/KEEP_Technology_Stack_Analysis.md**
- Use: **docs/technical/PIPELINE_VISUALIZATION_CURRENT.md**
- Reference: **docs/thesis/FINE_TUNING_STRATEGY.md**

### Chapter 4: Implementation & Evaluation
- Use: **docs/thesis/THESIS_READY_UNLABELED_EVALUATION_PLAN.md**
- Use: **docs/technical/PIPELINE_TEST_RESULTS.md**
- Use: **docs/technical/QC_EXECUTION_SUMMARY.md**
- Reference: **docs/technical/root_cause_low_accuracy.md**

### Chapter 5: Results & Discussion
- Use: **docs/thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md**
- Use: **docs/thesis/QA_LAB_TO_LIFE_GAP.md**
- Cite: **papers/domain_adaptation/**

### Chapter 6: Conclusion & Future Work
- Use: **docs/thesis/KEEP_Production_Robustness_Guide.md**
- Use: **FINAL_3_PATHWAYS_TO_COMPLETE_THESIS.md**

### Appendix
- Use: **docs/technical/** (all files)
- Reference: **src/README.md**

---

## 🔍 Finding Specific Content

### To Find Papers on...

| Topic | Location | Search For |
|-------|----------|------------|
| Anxiety detection methods | papers/anxiety_detection/ | ADAM, anxiety, wearable |
| Domain shift problems | papers/domain_adaptation/ | transfer, cross-position |
| MLOps pipelines | papers/mlops_production/ | deployment, monitoring |
| Confidence calibration | papers/uncertainty_confidence/ | calibration, OOD |

### To Find Documentation on...

| Topic | File |
|-------|------|
| 4-layer monitoring | docs/thesis/UNLABELED_EVALUATION.md |
| Wrist placement issue | docs/thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md |
| Why accuracy can't be measured | docs/thesis/CONCEPTS_EXPLAINED.md |
| Pipeline architecture | docs/technical/PIPELINE_VISUALIZATION_CURRENT.md |
| Test results | docs/technical/PIPELINE_TEST_RESULTS.md |
| Paper analysis | docs/research/RESEARCH_PAPERS_ANALYSIS.md |

---

## ✅ Organization Checklist

- [x] Created docs/thesis/ folder
- [x] Created docs/research/ folder
- [x] Created docs/technical/ folder
- [x] Created docs/archive/ folder
- [x] Created papers/ subfolders by topic
- [x] Moved thesis-critical files to docs/thesis/
- [x] Moved research analysis to docs/research/
- [x] Moved technical docs to docs/technical/
- [x] Moved old files to docs/archive/
- [ ] Organize papers by topic (manual sorting recommended)
- [x] Created this organization map

---

## 📝 Notes for Paper Organization

**Manual sorting recommended for papers/** because:
1. 200+ papers need topic classification
2. Some papers span multiple categories
3. Better to sort as you read for literature review

**Suggested workflow:**
1. Read paper abstract
2. Determine primary topic
3. Move to appropriate folder
4. Update papers_summary.xlsx with new location

**Keep "papers needs to read/" as inbox** for unsorted papers.

---

## 🚀 Next Steps

1. **For thesis writing:** Start with docs/thesis/ files — they're ready-to-use
2. **For technical details:** Check docs/technical/ for implementation specifics
3. **For citations:** Browse papers/ folders by topic
4. **For progress tracking:** Keep FINAL_3_PATHWAYS_TO_COMPLETE_THESIS.md updated

**All markdown files preserved — nothing deleted!**
