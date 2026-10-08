import json
from pathlib import Path

import pytest
import torch
pytest.importorskip("torchvision")
from paired_discriminator.cifar_classmatched import models, class_indices, draw_matched_indices, train


@pytest.fixture
def config():
    c = json.loads((Path(__file__).parents[1] / "configs/cifar10-classmatched.json").read_text())
    return {**c, "width": 4, "latent_dim": 8, "batch_size": 12, "log_every": 1, "checkpoint_every": 2}


def test_matching_and_balance():
    labels = torch.arange(10).repeat(4)
    groups = class_indices(labels, 10)
    ids, requested = draw_matched_indices(groups, 128, torch.Generator().manual_seed(3), torch.Generator().manual_seed(4))
    torch.testing.assert_close(labels[ids], requested, rtol=0, atol=0)
    counts = torch.bincount(requested)
    assert counts.min() == 12 and counts.max() == 13
    with pytest.raises(ValueError, match="equal nonempty"):
        class_indices(torch.tensor([0, 0, 1]), 2)


def test_initialization_and_condition_input(config):
    vg, vd = models(config, "vanilla", 0)
    for method in ("rsgan", "paired"):
        g, d = models(config, method, 0)
        for key, value in vg.state_dict().items():
            torch.testing.assert_close(value, g.state_dict()[key], rtol=0, atol=0)
        if method == "rsgan":
            for a, b in zip(vd.parameters(), d.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    vg.eval()
    z = torch.randn(2, 8)
    with torch.no_grad():
        a, b = vg(z, torch.zeros(2, dtype=torch.long)), vg(z, torch.ones(2, dtype=torch.long))
    assert a.shape == (2, 3, 32, 32)
    assert not torch.equal(a, b)


@pytest.mark.parametrize("method", ["vanilla", "rsgan", "paired"])
def test_conditional_resume(config, tmp_path, method):
    labels = torch.arange(10).repeat(2)
    data = torch.randint(256, (20, 3, 32, 32), dtype=torch.uint8, generator=torch.Generator().manual_seed(99))
    train(config, method, 0, 4, tmp_path / "full", data, labels, device="cpu")
    train(config, method, 0, 2, tmp_path / "split", data, labels, device="cpu")
    train(config, method, 0, 4, tmp_path / "split", data, labels, device="cpu")
    a = torch.load(tmp_path / "full/latest.pt", weights_only=False)
    b = torch.load(tmp_path / "split/latest.pt", weights_only=False)
    for key in ("generator", "discriminator", "rngs"):
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
    assert [(r['d_loss'], r['g_loss']) for r in a['rows']] == [(r['d_loss'], r['g_loss']) for r in b['rows']]
    with pytest.raises(ValueError, match="changed"):
        train(config, method, 0, 6, tmp_path / "split", data, labels.flip(0), device="cpu")
