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
import os

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


def _torch_worker_info():
    """`torch.utils.data.get_worker_info()`, or None outside a worker.

    Imported lazily so this module stays importable without torch, the same
    convention the rest of the file follows.
    """
    try:
        from torch.utils.data import get_worker_info
    except ImportError:
        return None
    return get_worker_info()


def default_num_workers(max_workers: int = 4, reserve: int = 2) -> int:
    """A conservative, portable `DataLoader(num_workers=...)` for this host.

    Sample generation is CPU-bound and single-threaded, so workers are the
    cheapest real speedup available (see `StreamMapDataset`). This picks a
    count that is safe to commit to a config file and still correct on a
    different machine -- deliberately NOT "all the cores":

    - **Never takes the whole machine.** `reserve` cores are left for the
      training process itself and for whatever else is running; on a shared
      login node or a workstation, saturating every core is antisocial, and
      on a cluster it is usually also wrong (see below). `max_workers` caps
      the count regardless of how large the machine is, since the benefit
      flattens out well before the core count does.
    - **Respects a cluster allocation rather than the physical machine.**
      A SLURM job given 4 CPUs on a 128-core node must not start 126
      workers. `SLURM_CPUS_PER_TASK` is honored when set, and
      `os.sched_getaffinity` (Linux) reports the cpuset/taskset the process
      is actually confined to, which containers and schedulers both use.
      `os.cpu_count()` -- which reports the whole machine and ignores all of
      that -- is only the last resort.
    - **Always overridable** via `STREAMGOGGLES_NUM_WORKERS`, so a job
      script can force a value (including 0) without code changes.

    Returns 0 when there is nothing to spare, which makes `DataLoader` load
    synchronously in the main process -- the safe default, not an error.
    """
    override = os.environ.get("STREAMGOGGLES_NUM_WORKERS")
    if override is not None:
        try:
            return max(0, int(override))
        except ValueError:
            logger.warning(
                "STREAMGOGGLES_NUM_WORKERS=%r is not an integer; ignoring it.",
                override,
            )

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            available = int(slurm_cpus)
        except ValueError:
            available = None
    else:
        available = None

    if available is None:
        if hasattr(os, "sched_getaffinity"):
            available = len(os.sched_getaffinity(0))
        else:
            available = os.cpu_count() or 1

    return max(0, min(max_workers, available - reserve))


class TransformedDataset:
    """Applies a transform to every item of a dataset, forwarding any other
    attribute access (e.g. `eval_mode`, `eval_grid`) to the wrapped dataset.

    Lives here rather than being defined where it is used because
    `DataLoader(num_workers>0)` must be able to *import* it. Under the
    "spawn" start method (the default on macOS and Windows) workers receive
    the dataset by pickle, and a class defined in a notebook cell or a
    `__main__` script cannot be unpickled there -- it fails with
    `AttributeError: Can't get attribute '...' on <module '__main__'>`,
    which is how this was found: enabling workers broke the training
    notebook outright.

    Attributes:
        base: the wrapped dataset.
        transform: callable applied to each item returned by `base`.
    """

    def __init__(self, base, transform):
        """Initialize.

        Parameters:
            base: dataset supporting `__len__`/`__getitem__`.
            transform: callable applied to every item.
        """
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        """Length of the wrapped dataset."""
        return len(self.base)

    def __getitem__(self, idx):
        """Return the wrapped dataset's item at `idx`, transformed."""
        return self.transform(self.base[idx])

    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped dataset.

        `base`/`transform` are refused explicitly rather than forwarded, and
        that guard is load-bearing, not defensive noise: unpickling creates
        the instance WITHOUT calling `__init__`, so `__dict__` is empty and
        pickle's own probing (`__setstate__`, `__reduce_ex__`, ...) lands
        here; forwarding would then look up `self.base`, which is itself
        missing, and recurse until `RecursionError`. Since DataLoader
        workers under "spawn" transfer this object by pickle, that would
        break exactly the case this class was moved here to support.
        """
        if name in ("base", "transform"):
            raise AttributeError(name)
        return getattr(self.base, name)


def configure_torch_threads(num_workers: int = 0, max_threads: int = 1) -> int:
    """Cap PyTorch's intra-op thread pool, and return what it was set to.

    Torch defaults to using **every** core for intra-op parallelism (8 on a
    10-core machine here), which is wrong twice over for this pipeline:

    - **It oversubscribes the machine.** With `num_workers` DataLoader
      processes already generating samples in parallel, a main process also
      claiming every core competes with them for the same CPUs. The model
      step is a small fraction of each sample's cost anyway (~24ms against
      ~130ms of generation, PLAN.md section 6.16), so the threads buy very
      little here while taking a lot.
    - **It crashed outright in this environment.** Two copies of
      `libomp.dylib` end up loaded in one process (torch's, and one arriving
      via the scientific stack), and torch's first convolution after that
      segfaults inside `__kmp_create_worker` -> `pthread_create` -- confirmed
      from the macOS crash report, and reproducible. With a single thread
      torch never creates an OpenMP worker, so it never takes that path.
      Duplicate OpenMP runtimes are an environment/packaging problem rather
      than something this project can fix, but a thread cap avoids it and is
      worth having on its own merits regardless.

    Parameters:
        num_workers: DataLoader worker count this run will use; reserved for
            callers that want to scale threads against it.
        max_threads: upper bound on torch's intra-op threads. Defaults to 1
            deliberately -- see above.

    Returns:
        The thread count actually set, or 0 if torch isn't installed.
    """
    try:
        import torch
    except ImportError:
        return 0

    override = os.environ.get("STREAMGOGGLES_TORCH_THREADS")
    if override is not None:
        try:
            threads = max(1, int(override))
        except ValueError:
            logger.warning(
                "STREAMGOGGLES_TORCH_THREADS=%r is not an integer; ignoring it.",
                override,
            )
            threads = max(1, max_threads)
    else:
        threads = max(1, max_threads)

    torch.set_num_threads(threads)
    return threads


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
      eval set must be reproducible across runs. With
      `build_eval_grid(..., n_replicates=k)`, each parameter combination
      appears k times with k different seeds: k independent realizations
      (different placement/window/orientation/noise) of the same physical
      parameters, so a per-parameter metric can be a mean and spread rather
      than one arbitrary draw. Each sample's own replicate index is recorded
      in `Sample.metadata["replicate"]`.

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
        # Set on first use inside a DataLoader worker; see
        # _ensure_worker_local_rng for why sharing one stream is unsafe.
        self._worker_id = None
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
        replicate = self.eval_grid.replicates[idx]
        full_params = {
            name: (spec.value if spec.is_fixed() else point[name])
            for name, spec in self.config.params.items()
        }

        # The store addresses samples by params, so replicates of the SAME
        # parameter combination (identical `full_params`, different seeds)
        # would all collide on one cache entry and return the first one
        # generated -- silently collapsing k realizations back into 1. The
        # replicate index therefore has to be part of the *addressing* key,
        # but NOT of the params handed to inject_single_stream, which only
        # ever takes real stream parameters.
        store_params = (
            full_params if replicate == 0 else {**full_params, "replicate": replicate}
        )

        def _generate(_params: dict) -> Sample:
            sample = self.injector.inject_single_stream(
                full_params, np.random.default_rng(seed)
            )
            if sample.metadata is not None:
                sample.metadata["replicate"] = replicate
            return sample

        return self.store.get_or_generate(store_params, _generate, persist=True)

    def _ensure_worker_local_rng(self) -> None:
        """Give each `DataLoader` worker its own RNG stream.

        Workers each receive a copy of this dataset -- forked or pickled
        depending on the platform's start method -- carrying `rng` in
        exactly the state the parent had. Without this, every worker spawns
        the same children and generates **identical samples**: measured 2
        unique samples out of 8 at `num_workers=4`, with no error and no
        slowdown to reveal it, which would silently train the model on
        duplicated data.

        Each worker deterministically takes its own child of the shared
        parent, so runs stay reproducible and workers stay independent.
        """
        worker_info = _torch_worker_info()
        if worker_info is None or self._worker_id == worker_info.id:
            return
        self.rng = self.rng.spawn(worker_info.num_workers)[worker_info.id]
        self._worker_id = worker_info.id

    def _get_training_sample(self, idx: int) -> Sample:
        if not 0 <= idx < self.steps_per_epoch:
            raise IndexError(
                f"idx {idx} out of range for steps_per_epoch={self.steps_per_epoch}"
            )
        self._ensure_worker_local_rng()
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
