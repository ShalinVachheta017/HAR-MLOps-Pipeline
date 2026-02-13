#!/usr/bin/env python3
"""
scripts/verify_repository.py — Repository completeness checker
================================================================

Run this in the NEW repository after all 10 commits to verify that
every file, directory, import, and config is in place.

Usage:
    python scripts/verify_repository.py
    python scripts/verify_repository.py --fix        # auto-create missing dirs
    python scripts/verify_repository.py --verbose     # show every check

Exit codes:
    0 — all checks pass
    1 — one or more checks failed
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify")

# ============================================================================
# EXPECTED STRUCTURE
# ============================================================================

REQUIRED_FILES = [
    # COMMIT 1 — foundation
    ".gitignore",
    "requirements.txt",
    "README.md",
    "pytest.ini",
    "pyproject.toml",
    "setup.py",
    "run_pipeline.py",

    # COMMIT 2 — entity
    "src/__init__.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",

    # COMMIT 3 — domain adaptation
    "src/domain_adaptation/__init__.py",
    "src/domain_adaptation/adabn.py",

    # COMMIT 4 — data components
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",

    # COMMIT 5 — inference components
    "src/components/model_inference.py",
    "src/components/model_evaluation.py",
    "src/components/post_inference_monitoring.py",
    "src/components/trigger_evaluation.py",

    # COMMIT 6 — retraining components
    "src/components/model_retraining.py",
    "src/components/model_registration.py",
    "src/components/baseline_update.py",

    # COMMIT 7 — pipeline
    "src/pipeline/__init__.py",
    "src/pipeline/production_pipeline.py",

    # COMMIT 8 — utils & CLI
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/utils/main_utils.py",

    # Core (cross-cutting)
    "src/core/__init__.py",
    "src/core/exception.py",
    "src/core/logger.py",
    "src/config.py",

    # COMMIT 9 — tests
    "tests/__init__.py",
    "tests/conftest.py",

    # COMMIT 10 — config, docker, scripts, docs
    "config/pipeline_config.yaml",
    "config/mlflow_config.yaml",
    "config/prometheus.yml",
    "config/requirements.txt",
    "docker/Dockerfile.training",
    "docker/Dockerfile.inference",
    "scripts/preprocess.py",
    "scripts/train.py",
    "scripts/monitor_setup.py",
    "docs/README.md",
]

REQUIRED_DIRS = [
    "src",
    "src/components",
    "src/entity",
    "src/domain_adaptation",
    "src/pipeline",
    "src/utils",
    "src/core",
    "tests",
    "config",
    "docker",
    "scripts",
    "docs",
    "logs",
    "data",
    "models",
]

IMPORT_CHECKS = [
    # (module_path, description)
    ("src.config", "Path constants"),
    ("src.entity.config_entity", "Config dataclasses"),
    ("src.entity.artifact_entity", "Artifact dataclasses"),
    ("src.core.exception", "PipelineException"),
    ("src.core.logger", "get_pipeline_logger"),
    ("src.utils.common", "Common utilities"),
    ("src.utils.main_utils", "ML utilities"),
    ("src.pipeline.production_pipeline", "ProductionPipeline"),
    ("src.domain_adaptation.adabn", "AdaBN module"),
]

COMPONENT_CLASSES = [
    ("src.components.data_ingestion", "DataIngestion"),
    ("src.components.data_validation", "DataValidation"),
    ("src.components.data_transformation", "DataTransformation"),
    ("src.components.model_inference", "ModelInference"),
    ("src.components.model_evaluation", "ModelEvaluation"),
    ("src.components.post_inference_monitoring", "PostInferenceMonitoring"),
    ("src.components.trigger_evaluation", "TriggerEvaluation"),
    ("src.components.model_retraining", "ModelRetraining"),
    ("src.components.model_registration", "ModelRegistration"),
    ("src.components.baseline_update", "BaselineUpdate"),
]


# ============================================================================
# CHECKER
# ============================================================================

class RepositoryVerifier:
    """Verify that the new repository has all required files and imports."""

    def __init__(self, root: Path, fix: bool = False, verbose: bool = False):
        self.root = root
        self.fix = fix
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors: list = []

    # ── helpers ────────────────────────────────────────────────────────

    def _ok(self, msg: str):
        self.passed += 1
        if self.verbose:
            logger.info("  [+]  %s", msg)

    def _fail(self, msg: str, critical: bool = True):
        self.failed += 1
        self.errors.append(msg)
        marker = "FAIL" if critical else "WARN"
        logger.warning("  [!]  %s  (%s)", msg, marker)

    def _warn(self, msg: str):
        self.warnings += 1
        logger.info("  [~]  %s  (WARN)", msg)

    # ── checks ────────────────────────────────────────────────────────

    def check_files(self):
        logger.info("\n--- FILE EXISTENCE CHECK (%d files) ---", len(REQUIRED_FILES))
        for rel in REQUIRED_FILES:
            p = self.root / rel
            if p.exists():
                self._ok(rel)
            else:
                self._fail(f"Missing file: {rel}")

    def check_dirs(self):
        logger.info("\n--- DIRECTORY CHECK (%d dirs) ---", len(REQUIRED_DIRS))
        for rel in REQUIRED_DIRS:
            d = self.root / rel
            if d.is_dir():
                self._ok(rel + "/")
            elif self.fix:
                d.mkdir(parents=True, exist_ok=True)
                self._ok(f"{rel}/  (CREATED)")
            else:
                self._fail(f"Missing directory: {rel}/")

    def check_imports(self):
        logger.info("\n--- IMPORT CHECK (%d modules) ---", len(IMPORT_CHECKS))
        for module, desc in IMPORT_CHECKS:
            try:
                importlib.import_module(module)
                self._ok(f"import {module}  ({desc})")
            except Exception as exc:
                self._fail(f"import {module}  →  {exc}")

    def check_components(self):
        logger.info("\n--- COMPONENT CLASS CHECK (%d components) ---", len(COMPONENT_CLASSES))
        for module, cls_name in COMPONENT_CLASSES:
            try:
                mod = importlib.import_module(module)
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    self._ok(f"{module}.{cls_name}")
                else:
                    self._fail(f"{cls_name} not found in {module}")
            except Exception as exc:
                self._fail(f"{module}.{cls_name}  →  {exc}")

    def check_cli(self):
        logger.info("\n--- CLI CHECK ---")
        run_pipeline = self.root / "run_pipeline.py"
        if run_pipeline.exists():
            content = run_pipeline.read_text(encoding="utf-8")
            for flag in ["--retrain", "--adapt", "--stages", "--advanced"]:
                if flag in content:
                    self._ok(f"run_pipeline.py has {flag}")
                else:
                    self._fail(f"run_pipeline.py missing {flag} flag")
        else:
            self._fail("run_pipeline.py not found")

    def check_test_files(self):
        logger.info("\n--- TEST FILE CHECK ---")
        tests_dir = self.root / "tests"
        if not tests_dir.is_dir():
            self._fail("tests/ directory missing")
            return
        test_files = list(tests_dir.glob("test_*.py"))
        count = len(test_files)
        if count >= 5:
            self._ok(f"Found {count} test files")
        elif count > 0:
            self._warn(f"Only {count} test files — expected 10+")
        else:
            self._fail("No test_*.py files in tests/")

    def check_data_readiness(self):
        logger.info("\n--- DATA READINESS CHECK ---")
        data_dir = self.root / "data"
        if not data_dir.is_dir():
            self._fail("data/ directory missing — you need to copy your dataset")
            return

        for sub in ("raw", "processed", "prepared"):
            d = data_dir / sub
            if d.is_dir():
                files = list(d.iterdir())
                if files:
                    self._ok(f"data/{sub}/  ({len(files)} files)")
                else:
                    self._warn(f"data/{sub}/ exists but is empty")
            else:
                self._warn(f"data/{sub}/ not found — create it before running pipeline")

        # Check for the main dataset
        labeled = data_dir / "raw" / "all_users_data_labeled.csv"
        if labeled.exists():
            size_mb = labeled.stat().st_size / (1024 * 1024)
            self._ok(f"Dataset found: all_users_data_labeled.csv ({size_mb:.1f} MB)")
        else:
            self._warn("Dataset not found: data/raw/all_users_data_labeled.csv")
            logger.info("       Copy your dataset before running the pipeline")

    def check_model_readiness(self):
        logger.info("\n--- MODEL READINESS CHECK ---")
        model = self.root / "models" / "pretrained" / "fine_tuned_model_1dcnnbilstm.keras"
        if model.exists():
            size_mb = model.stat().st_size / (1024 * 1024)
            self._ok(f"Pretrained model found ({size_mb:.1f} MB)")
        else:
            self._warn("Pretrained model not found: models/pretrained/fine_tuned_model_1dcnnbilstm.keras")
            logger.info("       Copy your model before running inference")

    def check_pyproject(self):
        logger.info("\n--- PYTHON PACKAGING CHECK ---")
        for f in ("pyproject.toml", "setup.py"):
            p = self.root / f
            if p.exists():
                self._ok(f"{f} present")
            else:
                self._fail(f"{f} missing — needed for `pip install -e .`")

    # ── run all ───────────────────────────────────────────────────────

    def run(self) -> bool:
        logger.info("=" * 60)
        logger.info("REPOSITORY VERIFICATION  —  %s", self.root)
        logger.info("=" * 60)

        self.check_dirs()
        self.check_files()
        self.check_pyproject()
        self.check_imports()
        self.check_components()
        self.check_cli()
        self.check_test_files()
        self.check_data_readiness()
        self.check_model_readiness()

        # ── Summary ──────────────────────────────────────────────────
        total = self.passed + self.failed
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS:  %d / %d passed   |   %d failed   |   %d warnings",
                     self.passed, total, self.failed, self.warnings)
        logger.info("=" * 60)

        if self.failed == 0:
            logger.info("ALL CHECKS PASSED — repository is complete!")
        else:
            logger.warning("\nMissing items:")
            for e in self.errors:
                logger.warning("  - %s", e)

            logger.info("\nTo fix missing directories, run:")
            logger.info("  python scripts/verify_repository.py --fix")

        return self.failed == 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify that the repository has all required files and imports",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Auto-create missing directories",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show passing checks too",
    )
    args = parser.parse_args()

    verifier = RepositoryVerifier(
        root=PROJECT_ROOT,
        fix=args.fix,
        verbose=args.verbose,
    )
    ok = verifier.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
