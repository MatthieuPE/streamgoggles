"""streamgoggles: ML pipeline for stellar stream detection in matched-filter maps.

This package provides:
- Configuration management (config.py)
- Parameter-addressed storage (storage.py)
- Matched-filter pixelization and projection (matched_filter.py)
- Window sampling and placement (windows.py)
- Background loading, with pluggable sources (background_sources.py, background.py)
- Stream generation and injection (stream_sources.py, injector.py)
- Target label rasterization (rasterize.py)
- Datasets, models, training, and evaluation.
"""

__version__ = "0.1.0"
