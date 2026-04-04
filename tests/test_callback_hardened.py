from unittest.mock import patch
from apex_aegis.trainer.callback import DefragCallback
from apex_aegis.utils import DefragConfig

def test_callback_lifecycle():
    """Verify callback lifecycle methods call monitor."""
    with patch("apex_aegis.trainer.callback.DefragMonitor") as mock_monitor_cls, \
         patch("apex_aegis.trainer.callback.DDPSyncManager") as mock_ddp_cls:
        
        mock_monitor = mock_monitor_cls.return_value
        mock_monitor.last_predicted_score = 0.8
        mock_monitor.stop.return_value = mock_monitor
        mock_monitor.stats.return_value = {"total_compactions": 0, "total_freed_mb": 0.0}
        
        callback = DefragCallback()
        
        callback.on_train_begin()
        assert mock_monitor.start.called
        
        callback.on_step_begin()
        assert mock_monitor.auto_record.called
        
        callback.on_step_end()
        assert mock_monitor.auto_record.call_count == 2
        assert callback._step_count == 1
        
        callback.on_train_end()
        assert mock_monitor.stop.called

def test_callback_ddp_sync_trigger():
    """Verify callback triggers compaction when DDP sync is enabled."""
    config = DefragConfig(ddp_sync=True)
    with patch("apex_aegis.trainer.callback.DefragMonitor") as mock_monitor_cls, \
         patch("apex_aegis.trainer.callback.DDPSyncManager") as mock_ddp_cls:
        
        mock_monitor = mock_monitor_cls.return_value
        mock_monitor.config = config
        mock_monitor.pending_compaction = True
        mock_monitor.last_predicted_score = 0.8
        
        mock_ddp = mock_ddp_cls.return_value
        mock_ddp.check_global_compaction.return_value = True
        
        callback = DefragCallback(config=config)
        callback.on_step_end()
        
        assert mock_ddp.check_global_compaction.called
        assert mock_monitor.compactor.defragment_tensors.called
        assert callback.monitor.pending_compaction == False

def test_callback_stats():
    """Verify callback stats aggregation."""
    with patch("apex_aegis.trainer.callback.DefragMonitor") as mock_monitor_cls, \
         patch("apex_aegis.trainer.callback.DDPSyncManager") as mock_ddp_cls:
        
        mock_monitor = mock_monitor_cls.return_value
        mock_monitor.stats.return_value = {"total_compactions": 5}
        
        mock_ddp = mock_ddp_cls.return_value
        mock_ddp.get_avg_overhead.return_value = 1.5
        
        callback = DefragCallback()
        s = callback.stats()
        assert s["total_compactions"] == 5
        assert s["ddp_sync_overhead_ms"] == 1.5
