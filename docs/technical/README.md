# 📚 Technical Documentation - Reading Guide

> **Purpose:** This folder contains technical implementation details, pipeline audits, and execution guides.

---

## 🗺️ Reading Order

### Start Here (Essential)
| # | File | Purpose | Importance |
|---|------|---------|------------|
| 1 | [PIPELINE_RERUN_GUIDE.md](PIPELINE_RERUN_GUIDE.md) | **How to run the entire pipeline** - Commands for raw → inference | ⭐⭐⭐ CRITICAL |
| 2 | [pipeline_audit_map.md](pipeline_audit_map.md) | Visual map of all pipeline components and their connections | ⭐⭐⭐ CRITICAL |
| 3 | [QC_EXECUTION_SUMMARY.md](QC_EXECUTION_SUMMARY.md) | Quality control results and validation status | ⭐⭐ HIGH |

### Debugging & Troubleshooting
| # | File | Purpose | Importance |
|---|------|---------|------------|
| 4 | [root_cause_low_accuracy.md](root_cause_low_accuracy.md) | Analysis of accuracy issues and fixes | ⭐⭐ HIGH |
| 5 | [PIPELINE_TEST_RESULTS.md](PIPELINE_TEST_RESULTS.md) | Test execution logs and results | ⭐⭐ HIGH |

### Deep Dives (Read When Needed)
| # | File | Purpose | Importance |
|---|------|---------|------------|
| 6 | [evaluation_audit.md](evaluation_audit.md) | Detailed evaluation metrics and analysis | ⭐ REFERENCE |
| 7 | [tracking_audit.md](tracking_audit.md) | MLflow tracking configuration audit | ⭐ REFERENCE |
| 8 | [PIPELINE_VISUALIZATION_CURRENT.md](PIPELINE_VISUALIZATION_CURRENT.md) | Current state diagrams | ⭐ REFERENCE |

---

## 📖 Suggested Reading Path

```
For Running Pipeline:
    PIPELINE_RERUN_GUIDE.md → pipeline_audit_map.md → QC_EXECUTION_SUMMARY.md

For Debugging Issues:
    root_cause_low_accuracy.md → PIPELINE_TEST_RESULTS.md → evaluation_audit.md

For Understanding Architecture:
    pipeline_audit_map.md → PIPELINE_VISUALIZATION_CURRENT.md → tracking_audit.md
```

---

## ⏱️ Time Estimates

| File | Reading Time | When to Read |
|------|--------------|--------------|
| PIPELINE_RERUN_GUIDE | 10 min | Before running pipeline |
| pipeline_audit_map | 15 min | First time setup |
| QC_EXECUTION_SUMMARY | 5 min | After each pipeline run |
| root_cause_low_accuracy | 20 min | When accuracy drops |
| Others | 10-15 min each | As needed |

---

*Last Updated: January 20, 2026*
