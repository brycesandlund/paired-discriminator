import numpy as np
from paired_discriminator.cifar_conditional_representation import conditional_summary


def test_balanced_wrong_labels_do_not_imply_adherence():
    y = np.arange(10).repeat(10)
    p = np.eye(10)[(y + 1) % 10]
    r = conditional_summary(p, y)
    assert r['class_tv'] == 0
    assert r['adherence'] == 0
    assert r['confidence']['0.9']['correct_fraction_all_samples'] == 0


def test_confident_correct_mass_uses_all_samples():
    y = np.tile(np.arange(10), 2)
    p = np.eye(10)[y].copy()
    p[10:] = p[10:] * .5 + .05
    r = conditional_summary(p, y)
    assert r['adherence'] == 1
    assert r['confidence']['0.9']['correct_fraction_all_samples'] == .5
    assert r['per_class_adherence'] == [1.] * 10
