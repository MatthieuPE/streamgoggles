"""PyTorch Dataset for stream map generation (on-the-fly training, persisted eval).

Rationale: Torch Dataset abstraction enables standard torch.utils.data.DataLoader
usage and enables both on-the-fly generation (training, no I/O bottleneck) and
persisted samples (eval, deterministic, reproducible). Duck-typed as a torch
Dataset -- only __len__/__getitem__, no actual torch.utils.data.Dataset
subclassing -- so this module stays importable without torch installed, the
same lazy-import convention storage.ModelStore uses for its own torch calls.

Wrapping this dataset in a real `torch.utils.data.DataLoader`: pass
`collate_fn=stream_map_collate_fn` (below) whenever `StreamConfig.
background_fraction > 0` -- torch's default collate_fn cannot batch this
dataset's items once background-only samples (params={}) get mixed with
stream samples (params={richness, morphology, ...}) in the same batch.
"""

import logging

import numpy as np

from streamgoggles.background import Background
from streamgoggles.config import EvalGrid, StreamConfig, build_eval_grid
from streamgoggles.injector import StreamInjector, inject_background_only
from streamgoggles.sample import Sample
from streamgoggles.storage import SimulationStore
from streamgoggles.windows import sample_random_window

logger = logging.getLogger(__name__)


def _sample_to_dict(sample: Sample) -> dict:
    """Convert a Sample to the plain dict a DataLoader's collate_fn expects
    (arrays/dicts, not a dataclass) -- see `stream_map_collate_fn` below for
    the collate_fn this dict actually needs when background_fraction > 0."""
    return {
        "map_stack": sample.map_stack,
        "label_stack": sample.label_stack,
        "valid_mask": sample.valid_mask,
        "params": sample.params,
        "metadata": sample.metadata,
    }


def stream_map_collate_fn(batch: list[dict]) -> dict:
    """DataLoader collate_fn for batches of `StreamMapDataset` items.

    torch's own default collate_fn recursively collates every dict key,
    requiring each sample's "params"/"metadata" dict to have identical keys
    across the whole batch -- but they don't: `inject_background_only`'s
    samples (drawn whenever `StreamConfig.background_fraction > 0`) carry
    `params={}`, while injected-stream samples carry richness/morphology/
    etc., so any batch mixing the two crashes `default_collate` with a bare
    `KeyError` on whichever param name it hits first. (Found by actually
    training with `background_fraction > 0` through a real `DataLoader` --
    training/plain_runner.py's tests happened to only exercise
    `background_fraction=0.0`, which never mixes the two shapes.)

    This collates `map_stack`/`label_stack`/`valid_mask` into batched
    tensors via torch's own `default_collate` (so dtype/behavior stays
    identical to the default path), and leaves `params`/`metadata` as
    plain per-sample lists instead of trying to merge them -- exactly what
    `training.plain_runner.PlainTrainer` needs, since it never reads
    `params`/`metadata` from a batch at all.

    Parameters:
        batch: list of dicts, each shaped like `StreamMapDataset.__getitem__`'s
            return value.

    Returns:
        dict with `map_stack`/`label_stack`/`valid_mask` as batched
        tensors, and `params`/`metadata` as lists (length `len(batch)`) of
        the original per-sample dicts, unmerged.
    """
    from torch.utils.data import default_collate

    tensor_keys = ("map_stack", "label_stack", "valid_mask")
    collated = {
        key: default_collate([item[key] for item in batch]) for key in tensor_keys
    }
    collated["params"] = [item["params"] for item in batch]
    collated["metadata"] = [item["metadata"] for item in batch]
    return collated


class StreamMapDataset:
    """Torch Dataset returning (map_stack, label_stack, valid_mask, params, metadata).

    Modes:
    - Training (eval_mode=False): sample free parameters on-the-fly and
      generate a fresh sample via `injector` on every `__getitem__` call.
      `idx` has no stable identity -- it's ignored beyond bounds-checking --
      so `__len__` (`steps_per_epoch`) is a nominal per-epoch count for
      `DataLoader`, not a bound on how many distinct samples exist.
    - Eval (eval_mode=True): walk `eval_grid` (built from `config` via
      `build_eval_grid` if not given explicitly). Each point has its own
      fixed seed, so `__getitem__(idx)` always returns the same sample --
      persisted through `store` regardless of `config.persist`, since an
      eval set must be reproducible across runs.

    Attributes:
        config: StreamConfig instance (params, background_fraction, persist).
            Its `richness_kind`/`label_policy` should match whatever
            `injector` was itself built with -- this class doesn't re-derive
            or check them, it just samples `config.params` and hands the
            result to `injector.inject_single_stream`.
        background: Background instance, shared across every sample (built
            once, outside this dataset -- injector.py's own "matched filters
            touch the background exactly once" guarantee depends on this).
        injector: StreamInjector instance.
        store: SimulationStore instance.
        eval_mode: bool, True for eval mode.
        eval_grid: EvalGrid (eval mode only).
        steps_per_epoch: nominal `__len__` for training mode.
        rng: np.random.Generator; each __getitem__ call spawns an
            independent child from it (`Generator.spawn`, the same pattern
            `streamobs.observed.StreamInjector.inject` uses for its own
            per-survey children), so per-item generation doesn't share
            mutable state across calls.

    Rationale: Single interface handles both training (efficient) and eval
    (deterministic). `background_fraction` is intentionally NOT a separate
    constructor parameter (unlike the original skeleton) -- `config`'s own
    `background_fraction` is already the single source of truth for it, so a
    second copy here could silently disagree with it.
    """

    def __init__(
        self,
        config: StreamConfig,
        background: Background,
        injector: StreamInjector,
        store: SimulationStore,
        eval_mode: bool = False,
        eval_grid: EvalGrid | None = None,
        steps_per_epoch: int = 10_000,
        rng: np.random.Generator | None = None,
    ):
        """Initialize dataset.

        Parameters:
            config: StreamConfig instance.
            background: Background instance.
            injector: StreamInjector instance.
            store: SimulationStore instance.
            eval_mode: If True, walk `eval_grid`, always persisting through
                `store`. If False, generate on-the-fly; persisted only if
                `config.persist` is True.
            eval_grid: EvalGrid to walk in eval mode. If None and
                `eval_mode=True`, built from `config` via `build_eval_grid`
                with its own defaults. Must be None when `eval_mode=False`
                (nothing to do with it there).
            steps_per_epoch: Nominal `__len__` for training mode. Ignored in
                eval mode.
            rng: np.random.Generator (if None, a default one is created).
                Training mode only -- eval mode's reproducibility comes from
                `eval_grid`'s own per-point seeds, not this.

        Raises:
            ValueError if `eval_grid` is given while `eval_mode=False`.
        """
        if not eval_mode and eval_grid is not None:
            raise ValueError("eval_grid is only used when eval_mode=True")

        self.config = config
        self.background = background
        self.injector = injector
        self.store = store
        self.eval_mode = eval_mode
        self.steps_per_epoch = steps_per_epoch
        self.rng = rng if rng is not None else np.random.default_rng()
        self.eval_grid = (
            (eval_grid if eval_grid is not None else build_eval_grid(config))
            if eval_mode
            else None
        )

    def __len__(self) -> int:
        """Return dataset size.

        Eval mode: number of points in `eval_grid`.
        Training mode: `steps_per_epoch` -- a nominal count, not a bound on
            distinct samples (generation is on-the-fly and unlimited).
        """
        if self.eval_mode:
            return len(self.eval_grid.points)
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> dict:
        """Return one sample.

        Parameters:
            idx: Sample index. Eval mode: indexes into `eval_grid` (stable
                identity -- the same idx always returns the same sample).
                Training mode: bounds-checked against `steps_per_epoch` only
                -- every call generates fresh, independently random
                parameters regardless of idx's value.

        Returns:
            dict with keys map_stack, label_stack, valid_mask (np.ndarray),
            params (dict), metadata (dict or None) -- see sample.Sample.

        Raises:
            IndexError if idx is out of `__len__()`'s range.
        """
        if self.eval_mode:
            sample = self._get_eval_sample(idx)
        else:
            sample = self._get_training_sample(idx)
        return _sample_to_dict(sample)

    def _get_eval_sample(self, idx: int) -> Sample:
        if not 0 <= idx < len(self.eval_grid.points):
            raise IndexError(
                f"idx {idx} out of range for eval_grid of size {len(self.eval_grid.points)}"
            )
        point = self.eval_grid.points[idx]
        seed = self.eval_grid.seeds[idx]
        full_params = {
            name: (spec.value if spec.is_fixed() else point[name])
            for name, spec in self.config.params.items()
        }

        def _generate(params: dict) -> Sample:
            return self.injector.inject_single_stream(
                params, np.random.default_rng(seed)
            )

        return self.store.get_or_generate(full_params, _generate, persist=True)

    def _get_training_sample(self, idx: int) -> Sample:
        if not 0 <= idx < self.steps_per_epoch:
            raise IndexError(
                f"idx {idx} out of range for steps_per_epoch={self.steps_per_epoch}"
            )
        (rng,) = self.rng.spawn(1)

        if rng.random() < self.config.background_fraction:
            pix = self.injector.pix
            size_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
            window = sample_random_window(
                self.background.footprint, pix.nside, size_deg=size_deg, rng=rng
            )
            sample = inject_background_only(self.background, window, pix)
        else:
            params = {
                name: spec.sample(rng) for name, spec in self.config.params.items()
            }
            sample = self.injector.inject_single_stream(params, rng)

        # Background-only samples have params={} (inject_background_only) --
        # no natural identity to key a persisted cache entry on, so only
        # stream samples get persisted here.
        if self.config.persist and sample.params:
            self.store.save_sample(sample, sample.params)

        return sample
