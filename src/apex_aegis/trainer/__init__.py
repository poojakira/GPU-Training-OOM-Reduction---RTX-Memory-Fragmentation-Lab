"""apex_aegis.trainer — Training pipeline, auto-instrumentation, and DDP support."""

from apex_aegis.trainer.callback import DefragCallback
from apex_aegis.trainer.auto_instrument import auto_instrument
from apex_aegis.trainer.ddp import DDPSyncManager
from apex_aegis.trainer.training_hook import TrainingHook

__all__ = ["DefragCallback", "auto_instrument", "DDPSyncManager", "TrainingHook"]
