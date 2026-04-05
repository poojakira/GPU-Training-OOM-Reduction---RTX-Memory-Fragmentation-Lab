# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - Upcoming
### Added
- Added `CODE_OF_CONDUCT.md`.
- Added `CHANGELOG.md`.
- Added `.editorconfig`.
- Added GitHub Issue Templates (`bug_report.yml`, `feature_request.yml`).
- Added `pull_request_template.md`.
- Added professional badges to `README.md`.

## [2.0.0] - 2026-04-03
### Added
- Production-grade PyTorch infrastructure layer for predictive GPU memory defragmentation.
- Custom Triton kernels for physical tensor repacking on HBM.
- 4-layer Transformer Encoder for fragmentation risk prediction.
- Real-time AeroGrid Dashboard for telemetry visualization.
- Prometheus/Grafana integration for cluster-wide observability.
- Support for GPT-2, BERT, and ResNet-50 workloads.

### Changed
- Improved memory utilization from 65.4% to 94.1%.
- Reduced OOM exceptions to zero on validated hardware (RTX-class).
- Optimized throughput by up to 50% on fragmented workloads.

## [1.0.0] - 2026-01-15
### Added
- Initial proof-of-concept for reactive GPU memory defragmentation.
- Basic CUDA memory hooking mechanism.
- Simple allocation pattern logging.
