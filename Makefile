# Makefile for Apex-Aegis: Predictive GPU Memory Defragmenter

.PHONY: help install test run run-benchmarks dashboard docker-build lint clean

# Default target
help:
	@echo "Apex-Aegis Infrastructure Makefile"
	@echo "---------------------------------"
	@echo "install         : Install dependencies and the apex_aegis package in editable mode"
	@echo "test            : Run the pytest suite"
	@echo "run             : Start the ML infra monitor (requires configs/config.yaml)"
	@echo "run-benchmarks  : Run the full memory fragmentation benchmark suite"
	@echo "dashboard       : Start the AeroGrid monitoring dashboard"
	@echo "docker-build    : Build the production-grade Docker image"
	@echo "lint            : Run ruff for style and quality checks"
	@echo "clean           : Remove temporary files and build artifacts"

install:
	pip install -e "."

test:
	pytest tests/

run:
	python run.py --config configs/config.yaml

run-benchmarks:
	python run_benchmark.py --config configs/config.yaml

dashboard:
	cd dashboard && npm run dev

docker-build:
	docker build -t apex-aegis-infra:latest .

lint:
	ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
	@echo "Cleaned environment."
