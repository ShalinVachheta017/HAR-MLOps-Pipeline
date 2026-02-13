"""
src/components — Each pipeline stage as a self-contained component class.

Pattern:
    component = SomeComponent(config, previous_artifact)
    artifact  = component.initiate_<stage_name>()

Components are thin wrappers around the existing standalone modules
(e.g. sensor_data_pipeline, data_validator, preprocess_data, …).
"""
