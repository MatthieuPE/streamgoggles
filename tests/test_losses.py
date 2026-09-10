"""Test loss functions.

All losses in models/losses.py accept optional valid_mask and soft float
targets. Coverage here: shape validation, correct masking (invalid pixels
truly excluded -- not down-weighted, verified by swapping masked-out
values for garbage and checking the loss is unchanged), hand-computed
values on tiny tensors for the analytic losses (Dice/Tversky/MSE/
WeightedMSE), agreement with torch's own primitives where one exists
(MSE, BCEWithLogits), the get_loss() factory, gradient flow, and one
end-to-end test combining a real models.unet.UNet with a loss.
"""

import pytest
import torch
import torch.nn.functional as F

from streamgoggles.models.losses import (
    BCEWithLogitsLoss,
    DiceLoss,
    FocalLoss,
    MSELoss,
    TverskyLoss,
    WeightedMSELoss,
    get_loss,
)
from streamgoggles.models.unet import UNet

pytestmark = pytest.mark.losses

ALL_LOSS_NAMES = ["dice", "focal", "tversky", "bce", "mse", "weighted_mse"]


def _loss_instance(name):
    return get_loss(name)


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_LOSS_NAMES)
def test_mismatched_pred_target_shape_raises(name):
    loss = _loss_instance(name)
    pred = torch.rand(1, 2, 4, 4)
    target = torch.rand(1, 2, 5, 5)
    with pytest.raises(ValueError, match="shape"):
        loss(pred, target)


@pytest.mark.parametrize("name", ALL_LOSS_NAMES)
def test_bad_ndim_raises(name):
    loss = _loss_instance(name)
    pred = torch.rand(2, 4, 4, 4, 4)
    target = torch.rand(2, 4, 4, 4, 4)
    with pytest.raises(ValueError, match="B, C, H, W"):
        loss(pred, target)


@pytest.mark.parametrize("name", ALL_LOSS_NAMES)
def test_bad_valid_mask_shape_raises(name):
    loss = _loss_instance(name)
    pred = torch.rand(1, 2, 4, 4)
    target = torch.rand(1, 2, 4, 4)
    bad_mask = torch.ones(3, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="valid_mask"):
        loss(pred, target, valid_mask=bad_mask)


@pytest.mark.parametrize("name", ALL_LOSS_NAMES)
def test_unbatched_chw_input_accepted(name):
    loss = _loss_instance(name)
    pred = torch.rand(2, 4, 4)
    target = torch.rand(2, 4, 4)
    out = loss(pred, target)
    assert out.shape == ()
    assert torch.isfinite(out)


@pytest.mark.parametrize("name", ALL_LOSS_NAMES)
def test_batched_valid_mask_accepted(name):
    loss = _loss_instance(name)
    pred = torch.rand(3, 2, 4, 4)
    target = torch.rand(3, 2, 4, 4)
    mask = torch.ones(3, 4, 4, dtype=torch.bool)
    mask[0, 0, 0] = False
    out = loss(pred, target, valid_mask=mask)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Masking correctness: true exclusion, not a weighted zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_LOSS_NAMES)
def test_masked_out_pixels_are_fully_excluded(name):
    loss = _loss_instance(name)
    torch.manual_seed(0)
    pred = torch.rand(1, 2, 4, 4)
    target = torch.rand(1, 2, 4, 4)
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[0, 0] = False
    mask[3, 3] = False

    baseline = loss(pred, target, valid_mask=mask)

    garbage_pred = pred.clone()
    garbage_target = target.clone()
    garbage_pred[:, :, 0, 0] = 1e6
    garbage_target[:, :, 0, 0] = -1e6
    garbage_pred[:, :, 3, 3] = -1e6
    garbage_target[:, :, 3, 3] = 1e6
    with_garbage = loss(garbage_pred, garbage_target, valid_mask=mask)

    assert torch.allclose(baseline, with_garbage, atol=1e-5)


@pytest.mark.parametrize("name", ALL_LOSS_NAMES)
def test_no_mask_equals_all_true_mask(name):
    loss = _loss_instance(name)
    torch.manual_seed(1)
    pred = torch.rand(1, 3, 4, 4)
    target = torch.rand(1, 3, 4, 4)
    all_true = torch.ones(4, 4, dtype=torch.bool)
    assert torch.allclose(loss(pred, target), loss(pred, target, valid_mask=all_true))


# ---------------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------------


def _tiny_pred_target():
    pred = torch.tensor([[0.9, 0.1], [0.2, 0.8]]).view(1, 1, 2, 2)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).view(1, 1, 2, 2)
    return pred, target


def test_dice_hand_computed_no_mask():
    pred, target = _tiny_pred_target()
    loss = DiceLoss()(pred, target)
    # intersection=1.7, pred_sum=2.0, target_sum=2.0 -> dice=3.4/4.0=0.85
    assert loss.item() == pytest.approx(0.15, abs=1e-4)


def test_dice_hand_computed_with_mask():
    pred, target = _tiny_pred_target()
    mask = torch.tensor([[True, True], [True, False]])
    loss = DiceLoss()(pred, target, valid_mask=mask)
    # excluding (1,1): intersection=0.9, pred_sum=1.2, target_sum=1.0
    # dice = 1.8/2.2
    expected = 1.0 - 1.8 / 2.2
    assert loss.item() == pytest.approx(expected, abs=1e-4)


def test_tversky_with_alpha_beta_half_equals_dice():
    pred, target = _tiny_pred_target()
    dice = DiceLoss()(pred, target)
    tversky = TverskyLoss(alpha=0.5, beta=0.5)(pred, target)
    assert tversky.item() == pytest.approx(dice.item(), abs=1e-4)


def test_tversky_higher_alpha_penalizes_false_positives_more():
    pred = torch.tensor([[0.9, 0.9]]).view(1, 1, 1, 2)
    target = torch.tensor([[1.0, 0.0]]).view(1, 1, 1, 2)
    low_alpha = TverskyLoss(alpha=0.1, beta=0.9)(pred, target)
    high_alpha = TverskyLoss(alpha=0.9, beta=0.1)(pred, target)
    # Second pixel (pred=0.9, target=0) is a false positive; a higher
    # alpha weights FP more heavily, so its loss should be larger.
    assert high_alpha.item() > low_alpha.item()


def test_mse_matches_torch_functional():
    torch.manual_seed(2)
    pred = torch.rand(2, 3, 5, 5)
    target = torch.rand(2, 3, 5, 5)
    ours = MSELoss()(pred, target)
    expected = F.mse_loss(pred, target)
    assert torch.allclose(ours, expected, atol=1e-6)


def test_mse_matches_torch_functional_with_mask():
    torch.manual_seed(3)
    pred = torch.rand(1, 2, 4, 4)
    target = torch.rand(1, 2, 4, 4)
    mask = torch.rand(4, 4) > 0.3
    ours = MSELoss()(pred, target, valid_mask=mask)
    full_mask = mask.view(1, 1, 4, 4).expand_as(pred)
    expected = ((pred - target) ** 2)[full_mask].mean()
    assert torch.allclose(ours, expected, atol=1e-6)


def test_bce_matches_torch_functional():
    torch.manual_seed(4)
    pred = torch.randn(2, 2, 4, 4) * 3
    target = torch.rand(2, 2, 4, 4)
    ours = BCEWithLogitsLoss()(pred, target)
    expected = F.binary_cross_entropy_with_logits(pred, target)
    assert torch.allclose(ours, expected, atol=1e-6)


def test_weighted_mse_normalized_equal_weights_matches_plain_mse():
    pred, target = torch.rand(1, 1, 2, 2), torch.ones(1, 1, 2, 2)
    weighted = WeightedMSELoss(normalize_weight=True)(pred, target)
    plain = MSELoss()(pred, target)
    # Uniform target=1 everywhere -> uniform weight -> normalized weighted
    # average degenerates to the plain mean.
    assert weighted.item() == pytest.approx(plain.item(), abs=1e-5)


def test_weighted_mse_hand_computed():
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 2, 2)
    target = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(1, 1, 2, 2)
    loss = WeightedMSELoss(normalize_weight=True)(pred, target)
    sq_err = (pred - target) ** 2  # [1, 1, 1, 1]
    weight = target / target.sum()  # [0, 1/6, 2/6, 3/6]
    expected = (weight * sq_err).sum()
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)


def test_weighted_mse_emphasizes_high_target_regions():
    # Same squared error everywhere, but target (hence weight) concentrated
    # on one pixel -> that pixel should dominate the loss.
    pred = torch.tensor([2.0, 2.0]).view(1, 1, 1, 2)
    target_concentrated = torch.tensor([100.0, 0.0]).view(1, 1, 1, 2)
    target_uniform = torch.tensor([1.0, 1.0]).view(1, 1, 1, 2)
    loss_concentrated = WeightedMSELoss()(pred, target_concentrated)
    loss_uniform = WeightedMSELoss()(pred, target_uniform)
    # error at pixel 0: (2-100)^2 huge; weight there ~1.0 when concentrated
    assert loss_concentrated.item() > loss_uniform.item()


# ---------------------------------------------------------------------------
# Focal loss: directional/sanity checks (no simple closed form to hand-check)
# ---------------------------------------------------------------------------


def test_focal_loss_smaller_for_confident_correct_predictions():
    confident_correct = torch.tensor([0.95, 0.05]).view(1, 1, 1, 2)
    confident_wrong = torch.tensor([0.05, 0.95]).view(1, 1, 1, 2)
    target = torch.tensor([1.0, 0.0]).view(1, 1, 1, 2)
    focal = FocalLoss()
    assert (
        focal(confident_correct, target).item() < focal(confident_wrong, target).item()
    )


def test_focal_loss_soft_targets_finite_and_between_hard_cases():
    pred = torch.tensor([0.5]).view(1, 1, 1, 1)
    focal = FocalLoss()
    loss_target_0 = focal(pred, torch.tensor([0.0]).view(1, 1, 1, 1))
    loss_target_1 = focal(pred, torch.tensor([1.0]).view(1, 1, 1, 1))
    loss_target_half = focal(pred, torch.tensor([0.5]).view(1, 1, 1, 1))
    assert torch.isfinite(loss_target_half)
    assert min(loss_target_0.item(), loss_target_1.item()) <= loss_target_half.item()
    assert loss_target_half.item() <= max(loss_target_0.item(), loss_target_1.item())


# ---------------------------------------------------------------------------
# Soft targets (not just hard 0/1) accepted by every loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_LOSS_NAMES)
def test_soft_targets_accepted_and_finite(name):
    loss = _loss_instance(name)
    torch.manual_seed(5)
    pred = torch.rand(2, 2, 4, 4)
    target = torch.rand(2, 2, 4, 4)  # soft, not just {0, 1}
    out = loss(pred, target)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_LOSS_NAMES)
def test_gradients_flow_to_pred(name):
    loss = _loss_instance(name)
    torch.manual_seed(6)
    pred = torch.rand(1, 2, 4, 4, requires_grad=True)
    target = torch.rand(1, 2, 4, 4)
    out = loss(pred, target)
    out.backward()
    assert pred.grad is not None
    assert torch.any(pred.grad != 0)


def test_masked_out_pred_receives_no_gradient():
    loss = MSELoss()
    pred = torch.rand(1, 1, 2, 2, requires_grad=True)
    target = torch.rand(1, 1, 2, 2)
    mask = torch.tensor([[True, False], [True, True]])
    out = loss(pred, target, valid_mask=mask)
    out.backward()
    assert pred.grad[0, 0, 0, 1].item() == 0.0


# ---------------------------------------------------------------------------
# get_loss() factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("dice", DiceLoss),
        ("focal", FocalLoss),
        ("tversky", TverskyLoss),
        ("bce", BCEWithLogitsLoss),
        ("mse", MSELoss),
        ("weighted_mse", WeightedMSELoss),
    ],
)
def test_get_loss_returns_correct_type(name, cls):
    assert isinstance(get_loss(name), cls)


def test_get_loss_passes_kwargs_through():
    loss = get_loss("focal", alpha=0.1, gamma=3.0)
    assert loss.alpha == 0.1
    assert loss.gamma == 3.0


def test_get_loss_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        get_loss("not_a_real_loss")


# ---------------------------------------------------------------------------
# End-to-end with a real UNet
# ---------------------------------------------------------------------------


def test_end_to_end_unet_and_weighted_mse():
    torch.manual_seed(7)
    model = UNet(in_channels=4, out_channels=4, base_width=4, depth=2, head="softplus")
    x = torch.rand(2, 4, 15, 15)
    target = torch.rand(2, 4, 15, 15) * 3  # e.g. stream-only star counts
    valid_mask = torch.ones(2, 15, 15, dtype=torch.bool)
    valid_mask[:, 0, 0] = False

    pred = model(x)
    loss_fn = get_loss("weighted_mse")
    loss = loss_fn(pred, target, valid_mask=valid_mask)
    loss.backward()

    assert torch.isfinite(loss)
    first_layer = model.encoder_blocks[0].block[0]
    assert first_layer.weight.grad is not None
    assert torch.any(first_layer.weight.grad != 0)
