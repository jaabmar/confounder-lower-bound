import numpy as np

from test_confounding.datasets.synthetic import RCT, Synthetic


def test_synthetic_init():
    synthetic_dataset = Synthetic(
        num_examples_obs=100,
        num_examples_rct=50,
        gamma_star=5.0,
        effective_conf=0.3,
        num_groups=2,
        theta=4.0,
        beta=0.75,
        sigma_y=0.1,
        domain=2.0,
        domain_rct=1.0,
        seed=1331,
        linear=True,
    )

    assert synthetic_dataset.num_examples == 100
    assert synthetic_dataset.num_examples_rct == 50
    assert synthetic_dataset.gamma_star == 5.0
    assert synthetic_dataset.effective_conf == 0.3
    assert synthetic_dataset.num_groups == 2
    assert synthetic_dataset.linear is True
    assert synthetic_dataset.x.shape == (100, 1)
    assert synthetic_dataset.t.shape == (100,)
    assert synthetic_dataset.y.shape == (100,)
    assert synthetic_dataset.tau.shape == (100,)
    assert synthetic_dataset.s.shape == (100,)


def test_synthetic_getitem():
    synthetic_dataset = Synthetic(
        num_examples_obs=100,
        num_examples_rct=50,
        gamma_star=5.0,
        effective_conf=0.3,
        num_groups=2,
        theta=4.0,
        beta=0.75,
        sigma_y=0.1,
        domain=2.0,
        domain_rct=1.0,
        seed=1331,
        linear=True,
    )
    index = 10  # Choose a valid index
    x, y = synthetic_dataset[index]

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.shape == (1,)
    assert y.shape == (1,)


def test_rct_creation():
    rct_dataset = RCT(num_examples=50, domain=1.0, sigma_y=0.1, theta=4.0, linear=True, seed=1331)

    assert rct_dataset.num_examples == 50
    assert rct_dataset.domain == 1.0
    assert rct_dataset.sigma_y == 0.1
    assert rct_dataset.theta == 4.0
    assert rct_dataset.linear is True
    assert rct_dataset.seed == 1331
    assert rct_dataset.x.shape == (50, 1)
    assert rct_dataset.t.shape == (50,)
    assert rct_dataset.y.shape == (50,)
    assert rct_dataset.pi.shape == (50,)
    assert rct_dataset.mu0.shape == (50,)
    assert rct_dataset.mu1.shape == (50,)
    assert rct_dataset.y0.shape == (50,)
    assert rct_dataset.y1.shape == (50,)
    assert rct_dataset.tau.shape == (50,)
    if rct_dataset.group_assigner is not None:
        assert rct_dataset.s.shape == (50,)
