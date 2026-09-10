"""Test models and architectures.

UNet (real torch, no mocking): forward shape for a variety of
in_channels/out_channels/base_width/depth combinations -- including small,
non-power-of-2 spatial sizes like this project's PixelizationSpec produces
(e.g. 15x15) -- head-type behavior (sigmoid in [0, 1], softplus >= 0,
identity unconstrained), encoder() bottleneck-feature shape, and
end-to-end gradient flow from output back to the first encoder layer.
"""

import pytest
import torch

from streamgoggles.models.unet import UNet

pytestmark = pytest.mark.models


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_unknown_head_raises():
    with pytest.raises(ValueError, match="Unknown head"):
        UNet(in_channels=2, out_channels=2, head="relu")


@pytest.mark.parametrize("head", ["sigmoid", "identity", "softplus"])
def test_valid_heads_construct(head):
    model = UNet(in_channels=2, out_channels=2, head=head)
    assert model.head_type == head


# ---------------------------------------------------------------------------
# forward() shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "base_width", "depth", "size"),
    [
        (4, 4, 8, 4, 16),  # per-channel output, matches label_stack shape
        (4, 1, 8, 4, 16),  # marginalized single-channel output
        (2, 2, 4, 1, 15),  # minimal depth
        (4, 4, 8, 4, 15),  # odd, non-power-of-2 spatial size
        (6, 6, 8, 3, 20),  # e.g. 2 distances x 3 filters
    ],
)
def test_forward_output_shape(in_channels, out_channels, base_width, depth, size):
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_width=base_width,
        depth=depth,
    )
    model.eval()
    x = torch.randn(2, in_channels, size, size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, out_channels, size, size)


def test_forward_handles_rectangular_input():
    model = UNet(in_channels=3, out_channels=3, base_width=4, depth=3)
    model.eval()
    x = torch.randn(1, 3, 17, 23)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 17, 23)


# ---------------------------------------------------------------------------
# Head types
# ---------------------------------------------------------------------------


def test_sigmoid_head_output_in_unit_interval():
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=2, head="sigmoid")
    model.eval()
    x = torch.randn(2, 2, 12, 12) * 5
    with torch.no_grad():
        out = model(x)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_softplus_head_output_non_negative():
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=2, head="softplus")
    model.eval()
    x = torch.randn(2, 2, 12, 12) * 5
    with torch.no_grad():
        out = model(x)
    assert torch.all(out >= 0.0)


def test_identity_head_can_produce_negative_values():
    # A freshly-initialized net with identity head has no output constraint;
    # bias the last layer negative to make sure nothing downstream clips it.
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=2, head="identity")
    with torch.no_grad():
        model.head_conv.bias.fill_(-10.0)
        model.head_conv.weight.zero_()
    model.eval()
    x = torch.randn(2, 2, 12, 12)
    with torch.no_grad():
        out = model(x)
    assert torch.all(out < 0.0)


def test_head_types_differ_on_same_input_weights():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 12, 12) * 5
    models = {}
    for head in ("sigmoid", "identity", "softplus"):
        torch.manual_seed(0)
        models[head] = UNet(
            in_channels=2, out_channels=2, base_width=4, depth=2, head=head
        )
        models[head].eval()

    with torch.no_grad():
        sigmoid_out = models["sigmoid"](x)
        identity_out = models["identity"](x)
        softplus_out = models["softplus"](x)

    # Same weights (same seed/architecture), only the head activation
    # differs, so identity must equal the pre-activation logits while
    # sigmoid/softplus are transformed versions of the exact same tensor.
    assert torch.allclose(sigmoid_out, torch.sigmoid(identity_out))
    assert torch.allclose(softplus_out, torch.nn.functional.softplus(identity_out))


# ---------------------------------------------------------------------------
# encoder()
# ---------------------------------------------------------------------------


def test_encoder_output_shape():
    model = UNet(in_channels=4, out_channels=4, base_width=8, depth=3)
    model.eval()
    x = torch.randn(2, 4, 24, 24)
    with torch.no_grad():
        features = model.encoder(x)
    expected_channels = 8 * (2**3)
    assert features.shape[0] == 2
    assert features.shape[1] == expected_channels
    # Three ceil-mode 2x poolings on a 24x24 input: exact power-of-2 case.
    assert features.shape[2] == 3
    assert features.shape[3] == 3


def test_encoder_downsamples_odd_size_without_crashing():
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=4)
    model.eval()
    x = torch.randn(1, 2, 15, 15)
    with torch.no_grad():
        features = model.encoder(x)
    assert features.shape[0] == 1
    assert features.shape[1] == 4 * (2**4)
    assert features.shape[2] > 0
    assert features.shape[3] > 0


def test_encoder_matches_forwards_internal_bottleneck():
    # encoder() should be a pure prefix of forward()'s own computation, not
    # a second, divergent code path.
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=2)
    model.eval()
    x = torch.randn(1, 2, 12, 12)
    with torch.no_grad():
        via_encoder = model.encoder(x)
        via_internal, _ = model._encode(x)
    assert torch.equal(via_encoder, via_internal)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_to_first_encoder_layer():
    model = UNet(in_channels=3, out_channels=3, base_width=4, depth=3, head="softplus")
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    first_layer = model.encoder_blocks[0].block[0]
    assert first_layer.weight.grad is not None
    assert torch.any(first_layer.weight.grad != 0)
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_gradients_flow_to_head_conv():
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=2, head="sigmoid")
    x = torch.randn(1, 2, 10, 10)
    out = model(x)
    out.sum().backward()
    assert model.head_conv.weight.grad is not None
    assert torch.any(model.head_conv.weight.grad != 0)
