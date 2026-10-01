"""Tests for the Hough line model (streamgoggles.models.hough)."""

import numpy as np
import pytest
import torch

from streamgoggles.models.hough import (
    HoughLines,
    HoughTargetTransform,
    HoughTransform,
    HoughUNet,
    hough_matrix,
    hough_target,
)


def line_image(size, theta_deg, rho, half_width=1.0):
    """A straight band x cos(theta) + y sin(theta) = rho, in pixels from the
    image centre."""
    y, x = np.mgrid[0:size, 0:size]
    x = x - (size - 1) / 2
    y = y - (size - 1) / 2
    theta = np.deg2rad(theta_deg)
    return (np.abs(x * np.cos(theta) + y * np.sin(theta) - rho) < half_width).astype(
        float
    )


def test_every_pixel_votes_once_per_theta():
    matrix, _, rhos = hough_matrix(20, 30, n_theta=12, rho_step=2.0)
    assert matrix.shape == (12 * rhos.size, 600)
    per_pixel = np.asarray(matrix.sum(axis=0)).ravel()
    np.testing.assert_allclose(per_pixel, 12.0)


@pytest.mark.parametrize("theta_deg, rho", [(30.0, 10.0), (100.0, -20.0), (0.0, 0.0)])
def test_a_line_peaks_at_its_own_cell(theta_deg, rho):
    grid = HoughLines(64, 64, n_theta=90, rho_step=2.0)
    sums = grid(line_image(64, theta_deg, rho))
    t, r = np.unravel_index(np.argmax(np.where(grid.valid, sums, -np.inf)), grid.shape)
    assert abs(np.rad2deg(grid.thetas[t]) - theta_deg) <= 2.0
    assert abs(grid.rhos[r] - rho) <= 2.0


def test_white_noise_gives_line_sums_of_order_one():
    grid = HoughLines(64, 64)
    noise = np.random.default_rng(0).normal(size=(64, 64))
    spread = grid(noise)[grid.valid].std()
    assert 0.6 < spread < 1.1


def test_short_lines_are_not_valid():
    grid = HoughLines(64, 64, min_pixels=20)
    assert np.all(grid.lengths[grid.valid] >= 20)
    assert not grid.valid.all()  # corner lines are too short


def test_target_marks_the_line_and_its_neighbours_only():
    grid = HoughLines(64, 64, n_theta=90, rho_step=2.0)
    target = hough_target(line_image(64, 45.0, 6.0), grid)
    t, r = np.nonzero(target)
    assert 1 <= target.sum() <= 30
    assert np.all(np.abs(np.rad2deg(grid.thetas[t]) - 45.0) <= 6.0)
    assert np.all(np.abs(grid.rhos[r] - 6.0) <= 6.0)


def test_empty_label_gives_no_line():
    grid = HoughLines(32, 32)
    assert hough_target(np.zeros((32, 32)), grid).sum() == 0


def test_torch_transform_matches_numpy():
    transform = HoughTransform(32, 32, n_theta=30, rho_step=2.0)
    image = np.random.default_rng(1).normal(size=(32, 32))
    tensor = torch.as_tensor(image, dtype=torch.float32)[None, None]
    np.testing.assert_allclose(
        transform(tensor)[0, 0].numpy(), transform.grid(image), rtol=1e-4, atol=1e-4
    )


def test_target_transform_replaces_label_and_valid_mask():
    grid = HoughLines(32, 32, n_theta=30)

    def view(sample):
        return dict(sample)

    sample = {
        "map_stack": np.zeros((7, 32, 32), np.float32),
        "label_stack": line_image(32, 60.0, 3.0)[None],
        "valid_mask": np.ones((32, 32), bool),
    }
    out = HoughTargetTransform(view, grid)(sample)
    assert out["label_stack"].shape == (1, *grid.shape)
    assert out["label_stack"].sum() > 0
    np.testing.assert_array_equal(out["valid_mask"], grid.valid)


def test_model_outputs_one_logit_per_line_and_trains():
    torch.manual_seed(0)
    model = HoughUNet(7, 32, features=4, depth=1, base_width=4, n_theta=30)
    x = torch.randn(2, 7, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, *model.hough.shape)
    out.sum().backward()
    assert model.backbone.head_conv.weight.grad is not None
    assert model.head_conv.weight.grad is not None


def test_the_input_reaches_the_lines_as_it_is():
    """The network over the lines receives the input's own line sums next to
    the backbone's: the matched filter's line sums are never lost."""
    model = HoughUNet(2, 32, features=3, depth=1, base_width=2, n_theta=30).eval()
    seen = {}
    model.lines.register_forward_hook(
        lambda module, inputs, output: seen.update(lines_input=inputs[0])
    )
    image = torch.as_tensor(line_image(32, 30.0, 4.0), dtype=torch.float32)
    x = torch.stack([image, torch.zeros_like(image)])[None]
    with torch.no_grad():
        model(x)
        expected = model.hough(x)
    lines_input = seen["lines_input"]
    assert lines_input.shape[1] == 3 + 2
    torch.testing.assert_close(lines_input[:, 3:], expected)
    sums = expected[0, 0].numpy()
    t, _ = np.unravel_index(np.argmax(sums), sums.shape)
    assert abs(np.rad2deg(model.hough.grid.thetas[t]) - 30.0) <= 4.0


def test_build_model_chooses_by_options():
    from streamgoggles.models import build_model
    from streamgoggles.models.unet import UNet

    default = build_model(7, depth=1, base_width=4)
    assert isinstance(default, UNet)
    assert default(torch.randn(1, 7, 16, 16)).shape == (1, 1, 16, 16)
    lines = build_model(7, 16, kind="hough", depth=1, base_width=4, n_theta=10)
    assert isinstance(lines, HoughUNet)
    with pytest.raises(ValueError):
        build_model(7, kind="hough")
    with pytest.raises(ValueError):
        build_model(7, kind="transformer")
