"""PyTorch Dataset for stream map generation (on-the-fly training, persisted eval).

Rationale: Torch Dataset abstraction enables standard torch.utils.data.DataLoader usage
and enables both on-the-fly generation (training, no I/O bottleneck) and persisted
samples (eval, deterministic, reproducible).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class StreamMapDataset:
    """Torch Dataset returning (map_stack, label_stack, valid_mask, metadata).
    
    Modes:
    - Training: sample free parameters on-the-fly, generate samples via injector.
    - Eval: walk enumerated eval grid, load persisted samples from SimulationStore.
    
    Attributes:
        config: StreamConfig instance.
        background: Background instance.
        injector: StreamInjector instance.
        store: SimulationStore instance.
        eval_mode: bool, True for eval mode.
        background_fraction: fraction of samples with no stream.
        rng: Random number generator.
    
    Rationale: Single interface handles both training (efficient) and eval (deterministic).
    Separate data collection on-the-fly vs. from cache based on mode.
    """
    
    def __init__(
        self,
        config,
        background,
        injector,
        store,
        eval_mode: bool = False,
        background_fraction: float = 0.0,
        rng = None
    ):
        """Initialize dataset.
        
        Parameters:
            config: StreamConfig instance.
            background: Background instance.
            injector: StreamInjector instance.
            store: SimulationStore instance.
            eval_mode: If True, load persisted samples (eval grid). If False, generate on-the-fly (training).
            background_fraction: Fraction [0, 1] of samples with no stream (pure background).
            rng: np.random.Generator (if None, default seed used).
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Return dataset size.
        
        Training mode: depends on how many epochs worth of data user wants,
            configurable via a config parameter (stub for now; default large).
        Eval mode: size of eval grid.
        """
        raise NotImplementedError
    
    def __getitem__(self, idx: int) -> dict:
        """Return one sample.
        
        Parameters:
            idx: Sample index.
        
        Returns:
            dict with keys:
                map_stack: (n_dist, ny, nx) float32
                label_stack: (n_dist, ny, nx) float32
                valid_mask: (ny, nx) bool
                params: dict
                metadata: dict or None
        
        Training mode: sample free parameters, generate via injector.
        Eval mode: load from store using eval_grid[idx].
        """
        raise NotImplementedError
