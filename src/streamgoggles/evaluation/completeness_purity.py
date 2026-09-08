"""Per-parameter completeness and purity curves.

Evaluate network on eval grid, extract per-stream detection decisions,
plot recovery vs each free parameter, compare network vs. baseline.

Rationale: Understand which parameters affect network performance and how,
enabling targeted improvements (e.g., "network fails at low surface brightness").
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_on_grid(
    model,
    eval_dataset: "StreamMapDataset",
    metrics: list[str],
    threshold: float = 0.5,
    device: str = "cpu"
) -> pd.DataFrame:
    """Walk eval grid, compute metrics per sample, join with parameters.
    
    Parameters:
        model: Trained model (torch.nn.Module).
        eval_dataset: StreamMapDataset in eval mode (persisted samples).
        metrics: List of metric names (e.g., ["dice", "iou", "precision"]).
        threshold: Threshold for binary metrics (default 0.5).
        device: "cpu" or "cuda".
    
    Returns:
        DataFrame with columns:
            (param_1, param_2, ..., param_N): parameter values
            (metric_1, metric_2, ...): computed metrics per sample
            id: sample identifier (for tracing)
    
    Raises:
        ValueError if metric names not recognized.
    """
    raise NotImplementedError


def plot_recovery_vs_parameter(
    results: pd.DataFrame,
    param_name: str,
    metric: str,
    ax = None,
    baseline_metric: str | None = None
) -> None:
    """Plot metric (y) vs parameter (x); overlay baseline if provided.
    
    Parameters:
        results: DataFrame from evaluate_on_grid().
        param_name: Parameter to plot on x-axis.
        metric: Metric to plot on y-axis.
        ax: matplotlib Axes (if None, create new).
        baseline_metric: Optional name of baseline metric column to overlay.
    
    Returns:
        None (plots to ax).
    
    Raises:
        KeyError if param_name or metric not in results.
    """
    raise NotImplementedError


def compute_completeness_purity(
    results: pd.DataFrame,
    stream_detection_fn,
    threshold: float = 0.5
) -> dict:
    """Compute per-stream completeness and purity.
    
    Completeness: fraction of true streams detected.
    Purity: fraction of detections that are true streams.
    
    Parameters:
        results: DataFrame from evaluate_on_grid().
        stream_detection_fn: Callable(metric_value) -> bool (detection decision).
            E.g., lambda x: x > 0.5 (detect if metric > threshold).
        threshold: Metric threshold (passed to stream_detection_fn if needed).
    
    Returns:
        dict with keys:
            completeness: float in [0, 1]
            purity: float in [0, 1]
            n_streams_total: int
            n_detected: int
    """
    raise NotImplementedError
