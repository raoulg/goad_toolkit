import numpy as np
import pytest
from scipy import stats

from goad_toolkit.analytics import DistributionFitter, FitResult
from goad_toolkit.distributions import DistributionRegistry


def test_registries_are_independent():
    first = DistributionRegistry()
    second = DistributionRegistry()

    first.register_distribution("pareto", stats.pareto, is_discrete=False, num_params=3)

    assert "pareto" in first.get_names()
    assert "pareto" not in second.get_names()
    assert "pareto" not in DistributionRegistry().get_names()


def test_registry_ships_the_default_families():
    names = DistributionRegistry().get_names()

    assert names == [
        "norm",
        "uniform",
        "lognorm",
        "poisson",
        "exponential",
        "skewnorm",
        "gamma",
        "weibull",
    ]


def test_unknown_distribution_raises():
    with pytest.raises(ValueError, match="not found in registry"):
        DistributionRegistry().get_distribution("pareto")


def test_fitter_uses_a_private_registry_by_default():
    fitter = DistributionFitter()
    DistributionRegistry().register_distribution(
        "pareto", stats.pareto, is_discrete=False, num_params=3
    )

    assert "pareto" not in fitter.registry


def test_fitter_accepts_a_registry():
    registry = DistributionRegistry()
    registry.distributions = {}
    registry.register_distribution("norm", stats.norm, is_discrete=False, num_params=2)

    fitter = DistributionFitter(registry)

    assert fitter.registry == ["norm"]


def test_fit_with_a_single_family_registry():
    registry = DistributionRegistry()
    registry.distributions = {}
    registry.register_distribution("norm", stats.norm, is_discrete=False, num_params=2)
    rng = np.random.default_rng(42)
    data = rng.normal(loc=3.0, scale=2.0, size=500)

    results = DistributionFitter(registry).fit(data, discrete=False)

    assert len(results) == 1
    fit = results[0]
    assert isinstance(fit, FitResult)
    assert fit.best_likelihood
    assert fit.params[0] == pytest.approx(3.0, abs=0.5)
