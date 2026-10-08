import json
from pathlib import Path

import torch

from paired_discriminator.image_flow import Velocity, interpolation, sample, train


def test_path_endpoints_and_velocity_target():
    a, b = torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4)
    x, v = interpolation(a, b, torch.tensor([0., 1.]))
    torch.testing.assert_close(x[0], a[0])
    torch.testing.assert_close(x[1], b[1])
    torch.testing.assert_close(v, b-a)


def test_midpoint_solves_time_linear_velocity_without_mutating_noise():
    class Field(torch.nn.Module):
        def forward(self, x, t):
            return (2*t)[:, None, None, None].expand_as(x)
    m = Field().train()
    noise = torch.randn(3, 1, 4, 4)
    before = noise.clone()
    torch.testing.assert_close(sample(m, noise, 4), noise+1)
    assert torch.equal(noise, before) and m.training


def test_native_shapes_and_time_gradient():
    torch.set_num_threads(1)
    for channels, size in [(1, 28), (3, 32)]:
        m = Velocity(channels, 8)
        torch.nn.init.normal_(m.out.weight, std=.01)
        y = m(torch.randn(2, channels, size, size), torch.tensor([.1, .9]))
        assert y.shape == (2, channels, size, size)
        y.square().mean().backward()
        assert m.time_net[0].weight.grad.abs().sum() > 0


def test_exact_resume_and_sampling_neutrality(tmp_path):
    config = json.loads((Path(__file__).parents[1]/'configs/image-flow.json').read_text())
    config.update(width=8, batch_size=2, checkpoint_every=2, log_every=2)
    data = torch.randint(256, (8, 1, 28, 28), dtype=torch.uint8)
    def observe(model, step, output):
        sample(model, torch.zeros(2, 1, 28, 28), 2)
    train(config, 'mnist', 0, 4, tmp_path/'full', data, 'cpu')
    train(config, 'mnist', 0, 2, tmp_path/'split', data, 'cpu', on_checkpoint=observe)
    train(config, 'mnist', 0, 4, tmp_path/'split', data, 'cpu', on_checkpoint=observe)
    a = torch.load(tmp_path/'full/latest.pt', weights_only=False)
    b = torch.load(tmp_path/'split/latest.pt', weights_only=False)
    for key in ('model', 'ema', 'rngs'):
        assert all(torch.equal(a[key][k], b[key][k]) for k in a[key])
    assert [r['loss'] for r in a['rows']] == [r['loss'] for r in b['rows']]
    for pid, state in a['optimizer']['state'].items():
        for key, value in state.items():
            assert torch.equal(value, b['optimizer']['state'][pid][key])
