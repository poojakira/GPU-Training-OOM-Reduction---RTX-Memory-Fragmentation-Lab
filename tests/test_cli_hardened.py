import sys
from unittest.mock import patch
from apex_aegis.cli import main

def test_cli_status_command(capsys):
    """Verify 'status' command outputs healthy indicators."""
    with patch("sys.argv", ["apex-aegis", "status"]), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA H100"), \
         patch("os.path.exists", return_value=True):
        
        main()
        captured = capsys.readouterr()
        assert "READY" in captured.out
        assert "NVIDIA H100" in captured.out

def test_cli_mock_telemetry_command(capsys):
    """Verify 'mock-telemetry' command runs without error (mocking time.sleep to exit)."""
    with patch("sys.argv", ["apex-aegis", "mock-telemetry", "--interval", "0.1"]), \
         patch("apex_aegis.defrag_engine.defragmenter.GPUMemoryDefragmenter._persist_telemetry") as mock_persist, \
         patch("time.sleep", side_effect=KeyboardInterrupt): # Exit loop immediately
        
        main()
        captured = capsys.readouterr()
        assert "Generating synthetic telemetry" in captured.out
        assert "Mock telemetry stopped" in captured.out

def test_cli_server_command():
    """Verify 'server' command starts uvicorn with correct args."""
    with patch("sys.argv", ["apex-aegis", "server", "--port", "9000"]), \
         patch("uvicorn.run") as mock_run:
        
        main()
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert kwargs["port"] == 9000
        assert "apex_aegis.api:app" in args

def test_cli_train_command():
    """Verify 'train' command invokes the trainer with correct config."""
    with patch("sys.argv", ["apex-aegis", "train", "--epochs", "5"]), \
         patch("apex_aegis.trainer.trainer.train") as mock_train:
        
        main()
        mock_train.assert_called_once()
        config = mock_train.call_args[1]["config"]
        assert config.train_epochs == 5

def test_cli_simulate_command():
    """Verify 'simulate' command launches the benchmark suite."""
    with patch("sys.argv", ["apex-aegis", "simulate", "--runs", "2"]), \
         patch("benchmarks.run_local_benchmark.main") as mock_bench:
        
        main()
        # Verify sys.argv was updated for the benchmark script
        assert sys.argv == ["run_local_benchmark.py", "--runs", "2", "--steps", "100"]
        mock_bench.assert_called_once()
