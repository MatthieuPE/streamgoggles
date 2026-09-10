"""Minimal training loop (plain PyTorch, no hyrax).

Used by the notebook and standalone scripts. Handles:
- epoch loop (forward, backward, optimize)
- checkpointing via ModelStore
- metric logging
- optional automatic mixed precision (AMP)
- simple learning rate scheduling (optional)

Rationale: Keeps training code simple and debuggable. No framework
magic; every step is explicit and understandable.

Batch format: `train_dl`/`val_dl` are expected to be `torch.utils.data.
DataLoader`s wrapping a `datasets.stream_map_dataset.StreamMapDataset`
(directly, or through `datasets.transforms.StreamMapTransform`), whose
`__getitem__` returns a dict with `map_stack`/`label_stack`/`valid_mask`
(plus `params`/`metadata`, untouched here) -- so each batch is itself a
dict of collated tensors under those same three keys, not a bare tuple.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from streamgoggles.storage import ModelStore

logger = logging.getLogger(__name__)


class PlainTrainer:
    """Minimal training loop for U-Net.

    Attributes:
        model: torch.nn.Module (e.g., UNet).
        optimizer: torch.optim.Optimizer.
        loss_fn: Loss function (e.g., DiceLoss).
        device: "cpu" or "cuda".
        amp: if True, use automatic mixed precision.

    Rationale: Explicit, single-threaded loop. No async loading, no parallel
    GPUs. Easy to understand, debug, and profile. For the first milestone.

    AMP implementation note: the skeleton this was built from sketched
    `torch.cuda.amp.GradScaler()` unconditionally when `amp=True`, which
    only makes sense on a CUDA device. Real development/testing here
    happens on CPU (this dev machine has no CUDA), so this uses the
    device-aware `torch.autocast(device_type=..., enabled=amp)` +
    `torch.amp.GradScaler(device_type, enabled=amp and device_type=="cuda")`
    APIs instead: autocast still applies (bfloat16) on CPU when amp=True,
    while the scaler -- only meaningful for CUDA fp16 -- stays a no-op
    there rather than erroring.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
        amp: bool = False,
    ):
        """Initialize trainer.

        Parameters:
            model: torch.nn.Module instance.
            optimizer: torch.optim.Optimizer.
            loss_fn: Loss function (should accept valid_mask).
            device: "cpu" or "cuda".
            amp: if True, use mixed precision.
        """
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.amp = amp
        self._device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        self.scaler = torch.amp.GradScaler(
            self._device_type, enabled=amp and self._device_type == "cuda"
        )

    def _run_batch(self, batch: dict, train: bool) -> tuple[torch.Tensor, int]:
        map_stack = batch["map_stack"].to(self.device, dtype=torch.float32)
        label_stack = batch["label_stack"].to(self.device, dtype=torch.float32)
        valid_mask = batch["valid_mask"].to(self.device, dtype=torch.bool)

        with torch.autocast(device_type=self._device_type, enabled=self.amp):
            pred = self.model(map_stack)
            loss = self.loss_fn(pred, label_stack, valid_mask)

        if train:
            self.optimizer.zero_grad()
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        return loss.detach(), pred, label_stack, valid_mask, map_stack.shape[0]

    def train_epoch(self, dl: DataLoader) -> dict:
        """Run one epoch on training data.

        Parameters:
            dl: DataLoader yielding sample dicts with map_stack, label_stack,
                valid_mask (see module docstring).

        Returns:
            dict with keys:
                loss: average loss (float), weighted by batch size
                n_batches: number of batches processed
                lr: current learning rate

        Rationale: Minimal logging; relies on caller for periodic validation/checkpointing.
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        n_batches = 0

        for batch in dl:
            loss, _, _, _, batch_size = self._run_batch(batch, train=True)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            n_batches += 1

        avg_loss = total_loss / n_samples if n_samples else float("nan")
        return {
            "loss": avg_loss,
            "n_batches": n_batches,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validate(self, dl: DataLoader, metrics_fn: Callable | None = None) -> dict:
        """Run one validation pass.

        Parameters:
            dl: DataLoader (eval set).
            metrics_fn: optional callable(pred, target, valid_mask) -> dict of metrics.

        Returns:
            dict with a ``loss`` key (average loss) plus, if `metrics_fn` is
            given, one key per metric it returns (a batch-size-weighted
            average of whatever scalar values `metrics_fn` returns).

        Rationale: No backprop; just inference and metric computation.
        """
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        metric_totals: dict[str, float] = {}

        with torch.no_grad():
            for batch in dl:
                loss, pred, label_stack, valid_mask, batch_size = self._run_batch(
                    batch, train=False
                )
                total_loss += loss.item() * batch_size
                n_samples += batch_size

                if metrics_fn is not None:
                    batch_metrics = metrics_fn(pred, label_stack, valid_mask)
                    for name, value in batch_metrics.items():
                        value = (
                            value.item()
                            if isinstance(value, torch.Tensor)
                            else float(value)
                        )
                        metric_totals[name] = (
                            metric_totals.get(name, 0.0) + value * batch_size
                        )

        result = {"loss": total_loss / n_samples if n_samples else float("nan")}
        for name, total in metric_totals.items():
            result[name] = total / n_samples if n_samples else float("nan")
        return result

    def train(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader | None = None,
        epochs: int = 10,
        model_store: ModelStore | None = None,
        config: dict | None = None,
        save_every: int = 1,
        callbacks: list | None = None,
    ) -> dict:
        """Full training loop: epochs, validation, checkpointing, logging.

        Parameters:
            train_dl: Training DataLoader.
            val_dl: Validation DataLoader (optional; if None, no validation).
            epochs: Number of epochs.
            model_store: ModelStore instance (optional; if provided, save checkpoints).
            config: Configuration dict (saved with checkpoint). Must contain a
                "model_config" key -- the exact kwargs to reconstruct
                `self.model` (`model.__class__(**config["model_config"])`,
                ModelStore.load_model's own contract) -- when `model_store`
                is given; the full dict is also used as-is as ModelStore's
                addressing `params` (mirroring the `{"model_config": ...,
                <other identifying fields>}` convention ModelStore's own
                tests use), so it can carry extra identifying fields (label
                policy, dataset config, ...) beyond just the model kwargs.
            save_every: Save checkpoint every N epochs (always saves the
                final epoch too, regardless of save_every).
            callbacks: Optional list of callable(epoch, trainer, metrics)
                hooks, called after each epoch with
                metrics={"train": train_epoch_metrics, "val": validate_metrics_or_None}.

        Returns:
            dict with keys:

            - ``train_losses``: list of per-epoch train loss
            - ``val_losses``: list of per-epoch val loss (or None if no validation)
            - ``final_checkpoint_path``: path to last saved checkpoint (or
              None if model_store not given)
            - ``epochs_run``: number of epochs trained

        Raises:
            ValueError if model_store is given but config is missing or
                doesn't contain "model_config".
        """
        if model_store is not None and (config is None or "model_config" not in config):
            raise ValueError(
                "config must be provided with a 'model_config' key when model_store is given"
            )

        train_losses = []
        val_losses = [] if val_dl is not None else None
        final_checkpoint_path: Path | None = None

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_dl)
            train_losses.append(train_metrics["loss"])

            val_metrics = None
            if val_dl is not None:
                val_metrics = self.validate(val_dl)
                val_losses.append(val_metrics["loss"])

            if callbacks:
                for callback in callbacks:
                    callback(epoch, self, {"train": train_metrics, "val": val_metrics})

            is_last_epoch = epoch == epochs - 1
            if model_store is not None and (
                (epoch + 1) % save_every == 0 or is_last_epoch
            ):
                metrics_snapshot = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"]
                    if val_metrics is not None
                    else None,
                }
                final_checkpoint_path = model_store.save_model(
                    self.model,
                    config=config["model_config"],
                    metrics=metrics_snapshot,
                    params=config,
                )

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_checkpoint_path": final_checkpoint_path,
            "epochs_run": epochs,
        }
