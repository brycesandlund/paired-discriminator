import numpy as np
from paired_discriminator.cifar_class_representation import summarize


def test_balanced_confident_and_collapsed():
    balanced = summarize(np.eye(10).repeat(100, axis=0))
    assert balanced['class_tv'] == 0
    assert balanced['covered_classes_1pct'] == 10
    assert balanced['confidence']['0.9']['augmented_tv'] == 0
    collapsed = summarize(np.eye(10)[np.zeros(1000, dtype=int)])
    assert np.isclose(collapsed['class_tv'], .9)
    assert collapsed['covered_classes_1pct'] == 1


def test_rejected_mass_is_not_renormalized():
    prob = np.concatenate([np.eye(10), np.full((10, 10), .1)])
    result = summarize(prob)['confidence']['0.9']
    np.testing.assert_allclose(result['accepted_mass'], .05)
    assert result['acceptance'] == .5
    assert np.isclose(result['augmented_tv'], .5)
