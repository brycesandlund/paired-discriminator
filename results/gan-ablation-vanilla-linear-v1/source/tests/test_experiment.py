import json

import pytest
import torch
from torch import nn

from paired_discriminator.experiment import (ROOT, centers, coverage_metrics, models,
    pair_inputs, discriminator_loss, generator_loss, sample_real)


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/ring8.json").read_text())


def test_metrics_uniform_collapsed_and_invalid(config):
    target = centers(config)
    uniform = coverage_metrics(target.repeat_interleave(100, dim=0), config)
    assert uniform["coverage"] == 8
    assert uniform["valid_fraction"] == 1
    assert uniform["mode_tv"] == 0
    collapsed = coverage_metrics(target[:1].repeat(800, 1), config)
    assert collapsed["coverage"] == 1
    assert collapsed["mode_tv"] == .875
    invalid = coverage_metrics(torch.zeros(800, 2), config)
    assert invalid["coverage"] == 0
    assert invalid["valid_fraction"] == 0
    assert invalid["mode_tv"] == 1


def test_coverage_requires_mass_not_one_lucky_point(config):
    samples = centers(config)[:1].repeat(1000, 1)
    samples[0] = centers(config)[1]
    assert coverage_metrics(samples, config)["coverage"] == 1


def test_sampler_matches_known_distribution(config):
    samples = sample_real(config, 100000, torch.Generator().manual_seed(42))
    metrics = coverage_metrics(samples, config)
    assert metrics["coverage"] == 8
    assert .985 < metrics["valid_fraction"] < .993
    assert metrics["mode_tv"] < .025


def test_generator_initialization_paired_across_methods(config):
    a, _ = models(config, "vanilla", 42)
    b, _ = models(config, "paired", 42)
    for first, second in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_slots_and_generator_gradient_both_orientations():
    real = torch.tensor([[2., 3.], [2., 3.]])
    fake = torch.tensor([[4., 5.], [4., 5.]], requires_grad=True)
    slots = torch.tensor([True, False])
    inputs = pair_inputs(real, fake, slots)
    torch.testing.assert_close(inputs, torch.tensor([[2., 3., 4., 5.], [4., 5., 2., 3.]]))
    d = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        d.weight.copy_(torch.tensor([[1., 0., -1., 0.]]))
    d.requires_grad_(False)
    loss = generator_loss(d, real, fake, "paired", slots)
    loss.backward()
    # Opposite labels and slots must produce identical gradients for identical fake points.
    torch.testing.assert_close(fake.grad[0], fake.grad[1])
    assert fake.grad[0, 0] < 0
    assert fake.grad[:, 1].abs().sum() == 0


def test_discriminator_step_does_not_backpropagate_into_generator():
    real = torch.randn(4, 2)
    fake = torch.randn(4, 2, requires_grad=True)
    for method, inputs in [("vanilla", 2), ("paired", 4), ("rsgan", 2)]:
        d = nn.Linear(inputs, 1)
        discriminator_loss(d, real, fake, method, torch.tensor([True, False, True, False])).backward()
        assert fake.grad is None
        assert d.weight.grad is not None


def test_rsgan_loss_and_gradient():
    d = nn.Linear(2, 1)
    with torch.no_grad():
        d.weight.copy_(torch.tensor([[1., 0.]]))
        d.bias.fill_(7.)
    real = torch.tensor([[2., 0.], [2., 0.]])
    fake = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
    slots = torch.tensor([True, False])
    dl = discriminator_loss(d, real, fake, "rsgan", slots)
    gl = generator_loss(d, real, fake, "rsgan", slots)
    torch.testing.assert_close(dl, torch.tensor(0.31326169))
    torch.testing.assert_close(gl, torch.tensor(1.31326169))
    gl.backward()
    torch.testing.assert_close(fake.grad[:, 0], torch.full((2,), -0.3655293))
    assert d.bias.grad.item() == 0  # A shared score offset must cancel.


def test_rsgan_matches_vanilla_initial_models(config):
    for a, b in zip(models(config, "vanilla", 2), models(config, "rsgan", 2)):
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
