import numpy as np

from src.estimands.utility import ap_skill, brier_skill, log_loss_skill, utility_estimands
from src.measurement_process.density import observed_hour_density


def test_utility_estimands():
    result = utility_estimands(0.80, 0.74, 0.76, 0.68)
    assert np.isclose(result["IUL"], 0.06)
    assert np.isclose(result["EUL"], 0.08)
    assert np.isclose(result["ITL"], 0.02)


def test_skill_transformations():
    prevalence = 0.2
    assert np.isclose(ap_skill(prevalence, prevalence), 0.0)
    assert np.isclose(brier_skill(prevalence * (1 - prevalence), prevalence), 0.0)
    null_ll = -(prevalence * np.log(prevalence) + (1-prevalence) * np.log(1-prevalence))
    assert np.isclose(log_loss_skill(null_ll, prevalence), 0.0)


def test_floor_hour_density_boundary():
    offsets = [0, 59, 60, 1439, 1440, -1]
    assert np.isclose(observed_hour_density(offsets), 3 / 24)
