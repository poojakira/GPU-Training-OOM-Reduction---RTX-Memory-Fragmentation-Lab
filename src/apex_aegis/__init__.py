"""
apex_aegis — Predictive GPU Memory Defragmenter
================================================

A Transformer-driven proactive CUDA memory optimizer for PyTorch.

Quick Start::

    from apex_aegis import auto_instrument

    # Wrap your model and optimizer with zero code changes
    model, optimizer = auto_instrument(model, optimizer)

    # ... standard training loop ...
"""

__version__ = "2.0.0"
__author__ = "GPU Defrag Infrastructure Team"

from apex_aegis.scheduler.monitor import DefragMonitor
from apex_aegis.trainer.callback import DefragCallback
from apex_aegis.trainer.auto_instrument import auto_instrument
from apex_aegis.trainer.ddp import DDPSyncManager
from apex_aegis.profiler.collector import AllocationCollector
from apex_aegis.scheduler.predictor import FragPredictor
from apex_aegis.defrag_engine.defragmenter import GPUMemoryDefragmenter

# Re-exported from migrated modules for unified namespace
from apex_aegis.profiler.allocator_logger import AllocatorLogger
from apex_aegis.scheduler.risk_model import OOMRiskModel
from apex_aegis.trainer.training_hook import TrainingHook
from apex_aegis.defrag_engine.policy import MitigationPolicy

__all__ = [
    "DefragMonitor",
    "DefragCallback",
    "auto_instrument",
    "DDPSyncManager",
    "AllocationCollector",
    "FragPredictor",
    "GPUMemoryDefragmenter",
    "AllocatorLogger",
    "OOMRiskModel",
    "TrainingHook",
    "MitigationPolicy",
]
