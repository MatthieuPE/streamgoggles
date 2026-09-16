"""Per-parameter completeness and purity curves.

Evaluate network on eval grid, extract per-stream detection decisions,
plot recovery vs each free parameter, compare network vs. baseline.

Rationale: Understand which parameters affect network performance and how,
enabling targeted improvements (e.g., "network fails at low surface brightness").
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from streamgoggles.evaluation.metrics import (
    confusion_matrix,
    correlation,
    dice,
    iou,
    mse,
    precision_recall,
    weighted_recall,
)

if TYPE_CHECKING:
    from streamgoggles.datasets.stream_map_dataset import StreamMapDataset

logger = logging.getLogger(__name__)

_SINGLE_VALUE_METRICS = {
    "iou": iou,
    "dice": dice,
    "mse": lambda pred, target, mask, threshold: mse(pred, target, mask),
    "correlation": lambda pred, target, mask, threshold: correlation(
        pred, target, mask
    ),
    "weighted_recall": weighted_recall,
}
_PRECISION_RECALL_NAMES = ("precision", "recall")
# "confusion" expands into four columns (tp/fp/tn/fn) rather than one, the
# same way "precision"/"recall" share a single call.
_CONFUSION_NAME = "confusion"
_CONFUSION_COLUMNS = ("tp", "fp", "tn", "fn")
_KNOWN_METRICS = (
    set(_SINGLE_VALUE_METRICS) | set(_PRECISION_RECALL_NAMES) | {_CONFUSION_NAME}
)


def evaluate_on_grid(
    model,
    eval_dataset: "StreamMapDataset",
    metrics: list[str],
    threshold: float = 0.5,
    device: str = "cpu",
) -> pd.DataFrame:
    """Walk eval grid, compute metrics per sample, join with parameters.

    Parameters:
        model: Trained model (torch.nn.Module).
        eval_dataset: StreamMapDataset in eval mode (persisted samples).
        metrics: List of metric names (e.g., ["dice", "iou", "precision"]).
            ``"confusion"`` is special: it expands into four columns
            (``tp``/``fp``/``tn``/``fn``) rather than one, the same way
            ``"precision"``/``"recall"`` share a single call.
        threshold: Threshold for binary metrics (default 0.5).
        device: "cpu" or "cuda".

    Returns:
        DataFrame with columns:

        - ``param_1``, ``param_2``, ..., ``param_N``: parameter values
        - ``metric_1``, ``metric_2``, ...: computed metrics per sample
        - ``id``: sample identifier (for tracing) -- the eval_grid index,
          which StreamMapDataset guarantees is a stable identity (the same
          idx always returns the same sample).
        - ``replicate``: present only when the eval grid was built with
          replicates (`build_eval_grid(..., n_replicates=k)`). Rows sharing
          the same parameter values but different ``replicate`` are
          independent realizations of those same parameters -- see
          `aggregate_over_replicates` to reduce them to a mean and spread.

    Raises:
        ValueError if metric names not recognized, or eval_dataset is not
            in eval mode.
    """
    import torch

    if not getattr(eval_dataset, "eval_mode", False):
        raise ValueError("eval_dataset must be a StreamMapDataset with eval_mode=True")

    unknown = [name for name in metrics if name not in _KNOWN_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {unknown}; available: {sorted(_KNOWN_METRICS)}"
        )

    model = model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for idx in range(len(eval_dataset)):
            sample = eval_dataset[idx]
            map_stack = np.asarray(sample["map_stack"], dtype=np.float32)
            label_stack = np.asarray(sample["label_stack"], dtype=np.float32)
            valid_mask = np.asarray(sample["valid_mask"], dtype=bool)

            x = torch.from_numpy(map_stack).unsqueeze(0).to(device)
            pred = model(x)[0].detach().cpu().numpy()

            row = dict(sample["params"])
            row["id"] = idx
            metadata = sample.get("metadata") or {}
            if "replicate" in metadata:
                row["replicate"] = metadata["replicate"]
            for name in metrics:
                if name in _PRECISION_RECALL_NAMES:
                    precision, recall = precision_recall(
                        pred, label_stack, valid_mask, threshold
                    )
                    row["precision"] = precision
                    row["recall"] = recall
                elif name == _CONFUSION_NAME:
                    row.update(
                        confusion_matrix(pred, label_stack, valid_mask, threshold)
                    )
                else:
                    row[name] = _SINGLE_VALUE_METRICS[name](
                        pred, label_stack, valid_mask, threshold
                    )
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_over_replicates(
    results: pd.DataFrame,
    metrics: list[str],
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """Reduce an `evaluate_on_grid` result with replicates to mean/std/count.

    A grid built with `build_eval_grid(..., n_replicates=k)` produces k rows
    per parameter combination -- k independent realizations (different
    placement/window/orientation/noise) of the same physical parameters.
    This collapses them into one row per combination, reporting each
    metric's mean, standard deviation and contributing-sample count.

    The spread is the point, not a formality: a per-parameter metric from a
    single realization can land anywhere within it, so `mean ± std` is what
    a "Dice at this richness" number should actually be read as. NaN metric
    values (evaluation.metrics' undefined-ratio convention) are excluded
    per metric rather than poisoning the whole group, and ``<metric>_n``
    records how many realizations actually contributed -- a mean over 2 of 5
    replicates is a different claim from a mean over 5 of 5.

    Parameters:
        results: DataFrame from `evaluate_on_grid`.
        metrics: metric column names to aggregate (e.g. ["dice", "iou"]).
        group_by: parameter columns defining a combination. Defaults to
            every column that is neither a requested metric nor bookkeeping
            (``id``/``replicate``) -- i.e. the sample's own parameters.

    Returns:
        DataFrame with one row per parameter combination: the `group_by`
        columns, plus ``<metric>_mean``/``<metric>_std``/``<metric>_n`` for
        each requested metric. ``std`` is NaN where only one realization
        contributed (undefined, not zero).

    Raises:
        KeyError if any requested metric column is missing from `results`.
    """
    missing = [name for name in metrics if name not in results.columns]
    if missing:
        raise KeyError(
            f"metric column(s) {missing} not in results columns {list(results.columns)}"
        )

    if group_by is None:
        excluded = set(metrics) | {"id", "replicate"}
        group_by = [name for name in results.columns if name not in excluded]
    if not group_by:
        raise ValueError(
            "No columns left to group by; pass group_by explicitly "
            "(e.g. group_by=['richness'])."
        )

    aggregated = (
        results.groupby(group_by, dropna=False)
        .agg({name: ["mean", "std", "count"] for name in metrics})
        .reset_index()
    )
    aggregated.columns = [
        column if not suffix else f"{column}_{'n' if suffix == 'count' else suffix}"
        for column, suffix in aggregated.columns
    ]
    return aggregated


def plot_recovery_vs_parameter(
    results: pd.DataFrame,
    param_name: str,
    metric: str,
    ax=None,
    baseline_metric: str | None = None,
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
        KeyError if param_name, metric, or baseline_metric not in results.
    """
    for name in (param_name, metric, *((baseline_metric,) if baseline_metric else ())):
        if name not in results.columns:
            raise KeyError(f"{name!r} not in results columns {list(results.columns)}")

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    sorted_results = results.sort_values(param_name)
    ax.plot(
        sorted_results[param_name], sorted_results[metric], marker="o", label=metric
    )
    if baseline_metric is not None:
        ax.plot(
            sorted_results[param_name],
            sorted_results[baseline_metric],
            marker="s",
            linestyle="--",
            label=baseline_metric,
        )
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric)
    ax.legend()


def compute_completeness_purity(
    results: pd.DataFrame,
    stream_detection_fn,
    metric: str,
    threshold: float = 0.5,
) -> dict:
    """Compute per-stream completeness and purity.

    Completeness: fraction of true streams detected.
    Purity: fraction of detections that are true streams.

    Parameters:
        results: DataFrame from evaluate_on_grid(). If it has an
            "is_true_stream" bool column, rows split into true streams
            (True) and non-stream/background rows (False) for a genuine
            purity calculation. Without that column, every row is treated
            as a true stream -- matching a StreamMapDataset eval grid,
            which (unlike training mode's background_fraction) always
            injects a real stream per point (see
            StreamMapDataset._get_eval_sample) -- so purity trivially
            reduces to 1.0 whenever anything is detected (there are no
            non-stream rows available to contaminate it); to measure a
            non-trivial purity, concatenate evaluate_on_grid() output from
            a stream eval grid with a separately-built background-only
            DataFrame carrying `is_true_stream=False`.
        stream_detection_fn: Callable(metric_value) -> bool (detection
            decision), e.g. `lambda x: x > 0.5`.
        metric: name of the `results` column `stream_detection_fn` is
            applied to, per row. (The skeleton this was implemented from
            didn't specify which column feeds `stream_detection_fn` --
            this parameter fixes that real gap; without it, the function
            has no way to know what to threshold.)
        threshold: Documented here for convenience when building
            `stream_detection_fn` (e.g. `lambda x: x > threshold`); not
            applied internally -- `stream_detection_fn` is the sole
            decision function this method calls.

    Returns:
        dict with keys:

        - ``completeness``: float in [0, 1] (NaN if there are no true streams)
        - ``purity``: float in [0, 1] (NaN if nothing was detected)
        - ``n_streams_total``: int, number of true-stream rows
        - ``n_detected``: int, number of rows stream_detection_fn flagged
          (across both true streams and, if present, non-stream rows)

    Raises:
        KeyError if `metric` not in results.
    """
    if metric not in results.columns:
        raise KeyError(
            f"metric {metric!r} not in results columns {list(results.columns)}"
        )

    detected = results[metric].apply(stream_detection_fn)

    if "is_true_stream" in results.columns:
        true_stream_mask = results["is_true_stream"].astype(bool)
    else:
        true_stream_mask = pd.Series(True, index=results.index)

    n_streams_total = int(true_stream_mask.sum())
    n_detected_true = int((detected & true_stream_mask).sum())
    n_detected_total = int(detected.sum())

    completeness = (
        n_detected_true / n_streams_total if n_streams_total > 0 else float("nan")
    )
    purity = (
        n_detected_true / n_detected_total if n_detected_total > 0 else float("nan")
    )

    return {
        "completeness": completeness,
        "purity": purity,
        "n_streams_total": n_streams_total,
        "n_detected": n_detected_total,
    }
