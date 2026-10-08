import json
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("torchvision")
from paired_discriminator.cifar import models, pair_inputs, loss_d, loss_g, train


@pytest.fixture
def config():
    c = json.loads((Path(__file__).parents[1] / "configs/cifar10.json").read_text())
    return {**c, "width": 4, "latent_dim": 8, "batch_size": 4, "log_every": 1, "checkpoint_every": 2}


def test_models_and_initialization(config):
    vg, vd = models(config, "vanilla", 0)
    rg, rd = models(config, "rsgan", 0)
    pg, pd = models(config, "paired", 0)
    for other in (rg, pg):
        for a, b in zip(vg.state_dict().values(), other.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(vd.parameters(), rd.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    fake = vg(torch.randn(4, 8))
    assert fake.shape == (4, 3, 32, 32)
    assert vd(fake).shape == (4,)
    assert pd(torch.cat((fake, fake), dim=1)).shape == (4,)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in vd.modules())


class Score(nn.Module):
    def __init__(self, paired=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.paired = paired

    def forward(self, x):
        value = x[:, :3].mean((1, 2, 3))
        if self.paired:
            value = value - x[:, 3:].mean((1, 2, 3))
        return self.scale * value


def test_pair_slots_loss_and_gradient():
    real = torch.ones(2, 3, 4, 4) * 2
    fake = torch.ones_like(real, requires_grad=True)
    slots = torch.tensor([True, False])
    paired = pair_inputs(real, fake, slots)
    torch.testing.assert_close(paired[0, :3], real[0])
    torch.testing.assert_close(paired[1, :3], fake[1])
    for method in ("rsgan", "paired"):
        d = Score(method == "paired")
        dl = loss_d(d, real, fake, method, slots)
        torch.testing.assert_close(dl, torch.tensor(.31326169))
        dl.backward()
        assert fake.grad is None
        gl = loss_g(d, real, fake, method, slots)
        torch.testing.assert_close(gl, torch.tensor(1.31326169))
        gl.backward()
        assert (fake.grad < 0).all()
        torch.testing.assert_close(fake.grad[0], fake.grad[1])
        fake.grad = None


@pytest.mark.parametrize("method", ["vanilla", "rsgan", "paired"])
def test_resume_matches_uninterrupted(config, tmp_path, method):
    data = torch.randint(256, (16, 3, 32, 32), dtype=torch.uint8, generator=torch.Generator().manual_seed(99))
    train(config, method, 0, 4, tmp_path / "full", data, device="cpu")
    train(config, method, 0, 2, tmp_path / "resume", data, device="cpu")
    train(config, method, 0, 4, tmp_path / "resume", data, device="cpu")
    a = torch.load(tmp_path / "full/latest.pt", weights_only=False)
    b = torch.load(tmp_path / "resume/latest.pt", weights_only=False)
    for key in ("generator", "discriminator", "rngs"):
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
    for ar, br in zip(a["rows"], b["rows"]):
        assert ar["d_loss"] == br["d_loss"] and ar["g_loss"] == br["g_loss"]
    with pytest.raises(ValueError, match="changed"):
        train({**config, "learning_rate": .001}, method, 0, 6, tmp_path / "resume", data, device="cpu")
