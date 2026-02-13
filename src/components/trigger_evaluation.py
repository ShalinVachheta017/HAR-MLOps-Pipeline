"""
Component 7 – Trigger Evaluation

Wraps:  src/trigger_policy.py  →  TriggerPolicyEngine
"""

import logging
from pathlib import Path
from typing import Optional

from src.entity.config_entity import TriggerEvaluationConfig, PipelineConfig
from src.entity.artifact_entity import (
    PostInferenceMonitoringArtifact,
    TriggerEvaluationArtifact,
)

logger = logging.getLogger(__name__)


class TriggerEvaluation:
    """Evaluate whether retraining should be triggered."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: TriggerEvaluationConfig,
        monitoring_artifact: PostInferenceMonitoringArtifact,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.monitoring_artifact = monitoring_artifact

    # ------------------------------------------------------------------ #
    def initiate_trigger_evaluation(self) -> TriggerEvaluationArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 7 — Trigger Evaluation")
        logger.info("=" * 60)

        from src.trigger_policy import (
            TriggerPolicyEngine,
            TriggerThresholds,
            CooldownConfig,
        )

        thresholds = TriggerThresholds(
            confidence_warn=self.config.confidence_warn,
            confidence_critical=self.config.confidence_critical,
            psi_warn=self.config.drift_psi_warn,
            psi_critical=self.config.drift_psi_critical,
        )
        cooldown = CooldownConfig(
            retrain_cooldown_hours=self.config.cooldown_hours,
        )
        state_dir = self.config.state_dir or (
            self.pipeline_config.logs_dir / "trigger"
        )
        state_file = Path(state_dir) / "trigger_state.json"

        engine = TriggerPolicyEngine(
            thresholds=thresholds,
            cooldown_config=cooldown,
            state_file=state_file,
        )

        decision = engine.evaluate(self.monitoring_artifact.monitoring_report)

        logger.info(
            "Trigger decision: action=%s  should_retrain=%s  alert=%s",
            decision.action, decision.should_trigger, decision.alert_level,
        )

        return TriggerEvaluationArtifact(
            should_retrain=decision.should_trigger,
            action=str(decision.action.value) if hasattr(decision.action, "value") else str(decision.action),
            alert_level=str(decision.alert_level.value) if hasattr(decision.alert_level, "value") else str(decision.alert_level),
            reasons=decision.recommendations if hasattr(decision, "recommendations") else [],
            cooldown_active=False,
        )
