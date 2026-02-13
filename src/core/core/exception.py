"""
Custom exception for HAR MLOps Pipeline.
Provides rich error context (file, line, stage) for debugging.
"""

import sys
import traceback


class PipelineException(Exception):
    """
    Custom exception that captures the originating file and line number.
    
    Usage:
        try:
            ...
        except Exception as e:
            raise PipelineException(e, sys) from e
    """

    def __init__(self, error: Exception, error_detail: object = None, stage: str = ""):
        super().__init__(str(error))
        self.stage = stage

        if error_detail is not None and hasattr(error_detail, "exc_info"):
            _, _, exc_tb = error_detail.exc_info()
            if exc_tb is not None:
                tb = traceback.extract_tb(exc_tb)
                last = tb[-1]
                self.message = (
                    f"[Stage: {stage}] Error in {last.filename} "
                    f"line {last.lineno}: {error}"
                )
            else:
                self.message = f"[Stage: {stage}] {error}"
        else:
            self.message = f"[Stage: {stage}] {error}"

    def __str__(self):
        return self.message
