"""
Tests for scripts/update_progress_dashboard.py
================================================
Validates TRS/ERS computation and markdown output format.
"""

import textwrap
from pathlib import Path

import pytest
import yaml

# Import the module under test
from scripts.update_progress_dashboard import (
    ascii_bar,
    checklist_summary,
    compute_ers,
    compute_trs,
    generate_dashboard,
    load_config,
    weeks_remaining,
)


# ---------------------------------------------------------------------------
# Sample YAML fixture
# ---------------------------------------------------------------------------

SAMPLE_YAML = textwrap.dedent("""\
    snapshot_date: "2026-02-12"
    deadline: "2026-05-20"

    sections:
      writing:
        percent: 13
        checklist:
          - item: "Ch 1 — Introduction"
            done: false
          - item: "Ch 2 — Literature Review"
            done: false
          - item: "Abstract written"
            done: false

      experiments:
        percent: 45
        checklist:
          - item: "Full pipeline on 26 sessions"
            done: false
          - item: "MLflow results exported"
            done: false
          - item: "Confusion matrix plots"
            done: true

      system:
        percent: 83
        checklist:
          - item: "docker-compose up runs"
            done: true
          - item: "All tests pass"
            done: true
          - item: "CI passes on push"
            done: false
          - item: "Prometheus deployed"
            done: false

    month_milestones:
      - month: 1
        label: "Data Ingestion"
        period: "Oct 2025"
        status: "done"
        percent: 95
        evidence:
          - "src/sensor_data_pipeline.py"
      - month: 2
        label: "Training"
        period: "Nov 2025"
        status: "done"
        percent: 82
        evidence:
          - "src/train.py"
      - month: 6
        label: "Thesis Writing"
        period: "Mar-Apr 2026"
        status: "in-progress"
        percent: 15
        evidence:
          - "docs/thesis/THESIS_STRUCTURE_OUTLINE.md"
""")


@pytest.fixture
def sample_data() -> dict:
    """Parse the sample YAML into a dict."""
    return yaml.safe_load(SAMPLE_YAML)


@pytest.fixture
def sample_yaml_file(tmp_path: Path) -> Path:
    """Write sample YAML to a temp file and return the path."""
    p = tmp_path / "progress.yaml"
    p.write_text(SAMPLE_YAML, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# TRS / ERS computation tests
# ---------------------------------------------------------------------------

class TestComputeTRS:
    """Tests for the Thesis Readiness Score formula."""

    def test_known_values(self):
        """TRS = 0.45*13 + 0.35*45 + 0.20*83 = 5.85 + 15.75 + 16.6 = 38.2"""
        result = compute_trs(13, 45, 83)
        assert abs(result - 38.2) < 0.1

    def test_zero_inputs(self):
        assert compute_trs(0, 0, 0) == 0.0

    def test_full_completion(self):
        result = compute_trs(100, 100, 100)
        assert abs(result - 100.0) < 0.01

    def test_writing_only(self):
        result = compute_trs(100, 0, 0)
        assert abs(result - 45.0) < 0.01

    def test_experiments_only(self):
        result = compute_trs(0, 100, 0)
        assert abs(result - 35.0) < 0.01

    def test_system_only(self):
        result = compute_trs(0, 0, 100)
        assert abs(result - 20.0) < 0.01


class TestComputeERS:
    """Tests for Engineering Readiness Score."""

    def test_matches_system(self):
        assert compute_ers(83) == 83

    def test_zero(self):
        assert compute_ers(0) == 0

    def test_full(self):
        assert compute_ers(100) == 100


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestWeeksRemaining:
    def test_known_range(self):
        assert weeks_remaining("2026-02-12", "2026-05-20") == 13

    def test_same_day(self):
        assert weeks_remaining("2026-05-20", "2026-05-20") == 0

    def test_past_deadline(self):
        assert weeks_remaining("2026-06-01", "2026-05-20") == 0


class TestAsciiBar:
    def test_zero(self):
        bar = ascii_bar(0, width=10)
        assert len(bar) == 10
        assert "\u2588" not in bar

    def test_full(self):
        bar = ascii_bar(100, width=10)
        assert len(bar) == 10
        assert "\u2591" not in bar

    def test_half(self):
        bar = ascii_bar(50, width=20)
        assert len(bar) == 20
        assert bar.count("\u2588") == 10

    def test_clamp_above_100(self):
        bar = ascii_bar(150, width=10)
        assert bar.count("\u2588") == 10

    def test_clamp_below_0(self):
        bar = ascii_bar(-10, width=10)
        assert bar.count("\u2588") == 0


class TestChecklistSummary:
    def test_mixed(self):
        items = [
            {"item": "a", "done": True},
            {"item": "b", "done": False},
            {"item": "c", "done": True},
        ]
        done, total = checklist_summary(items)
        assert done == 2
        assert total == 3

    def test_empty(self):
        done, total = checklist_summary([])
        assert done == 0
        assert total == 0

    def test_all_done(self):
        items = [{"item": "x", "done": True}]
        done, total = checklist_summary(items)
        assert done == 1
        assert total == 1


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid(self, sample_yaml_file: Path):
        data = load_config(sample_yaml_file)
        assert data["snapshot_date"] == "2026-02-12"
        assert "writing" in data["sections"]
        assert len(data["month_milestones"]) == 3

    def test_missing_key_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("snapshot_date: '2026-01-01'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required keys"):
            load_config(bad)


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------

class TestGenerateDashboard:
    def test_contains_headline_scores(self, sample_data: dict):
        md = generate_dashboard(sample_data)
        assert "Engineering Readiness Score (ERS)" in md
        assert "Thesis Readiness Score (TRS)" in md

    def test_contains_trs_value(self, sample_data: dict):
        md = generate_dashboard(sample_data)
        assert "38.2%" in md

    def test_contains_ers_value(self, sample_data: dict):
        md = generate_dashboard(sample_data)
        assert "83%" in md

    def test_contains_section_headers(self, sample_data: dict):
        md = generate_dashboard(sample_data)
        assert "## Section Progress" in md
        assert "## Checklists" in md
        assert "## Month Milestones" in md

    def test_contains_checklist_items(self, sample_data: dict):
        md = generate_dashboard(sample_data)
        assert "Ch 1" in md
        assert "docker-compose" in md
        assert "MLflow results" in md

    def test_contains_milestone_table(self, sample_data: dict):
        md = generate_dashboard(sample_data)
        assert "Data Ingestion" in md
        assert "Training" in md
        assert "Thesis Writing" in md

    def test_deterministic(self, sample_data: dict):
        """Running twice gives identical output."""
        md1 = generate_dashboard(sample_data)
        md2 = generate_dashboard(sample_data)
        assert md1 == md2

    def test_output_is_valid_markdown(self, sample_data: dict):
        """Basic check: starts with heading, contains tables."""
        md = generate_dashboard(sample_data)
        assert md.startswith("# Thesis Progress Dashboard")
        assert "|" in md  # tables present
        assert "---" in md  # horizontal rules present
