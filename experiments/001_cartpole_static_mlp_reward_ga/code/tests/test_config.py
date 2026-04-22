from awr.config import CartPoleGAConfig


def test_sigma_sigma_defaults_to_self_adaptive_value():
    config = CartPoleGAConfig()

    assert config.sigma_sigma == 0.1


def test_plateau_stop_patience_defaults_to_none():
    config = CartPoleGAConfig()

    assert config.plateau_stop_patience is None


def test_plateau_stop_smoothing_window_defaults_to_ten():
    config = CartPoleGAConfig()

    assert config.plateau_stop_smoothing_window == 10
