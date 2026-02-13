"""Pipeline module — production pipeline orchestration."""
from src.pipeline.production_pipeline import ProductionPipeline

# Keep old import available for backwards compat (deferred to avoid circular)
try:
    from src.pipeline.inference_pipeline import InferencePipeline
except ImportError:
    InferencePipeline = None
