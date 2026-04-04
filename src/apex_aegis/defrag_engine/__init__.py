"""apex_aegis.defrag_engine — GPU memory defragmentation engine."""

from apex_aegis.defrag_engine.defragmenter import GPUMemoryDefragmenter
from apex_aegis.defrag_engine.compactor import MemoryCompactor
from apex_aegis.defrag_engine.policy import MitigationPolicy

__all__ = ["GPUMemoryDefragmenter", "MemoryCompactor", "MitigationPolicy"]
