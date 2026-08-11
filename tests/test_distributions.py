import numpy as np
import pytest
from scipy import stats

from goad_toolkit.analytics import (
    DistributionFitter,
    FailedFit,
    FitResult,
    KSTestResult,
    fit_table,
)
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
        "bernoulli",
        "binomial",
        "nbinom",
        "beta",
    ]


def test_pareto_is_not_a_default_family():
    # Lesson 4 has the student register pareto themselves as the extensibility
    # exercise; shipping it by default would remove the point of that exercise.
    assert "pareto" not in DistributionRegistry().get_names()


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


def test_bernoulli_fits_a_small_p():
    # A single-parameter discrete bound used to floor at 0.1, which is a sane
    # floor for a count-scale rate (poisson's lambda) but clips a genuinely
    # small probability (bernoulli's p) to the floor instead of the true value.
    registry = DistributionRegistry()
    registry.distributions = {}
    registry.register_distribution(
        "bernoulli", stats.bernoulli, is_discrete=True, num_params=1
    )
    rng = np.random.default_rng(7)
    data = stats.bernoulli(p=0.05).rvs(3000, random_state=rng)

    results = DistributionFitter(registry).fit(data, discrete=True)

    fit = results[0]
    assert isinstance(fit, FitResult)
    assert fit.params[0] == pytest.approx(0.05, abs=0.02)


def test_loglikelihood_is_finite_for_discrete_families():
    # dist_obj.dist.logpdf does not exist on discrete scipy distributions
    # (they have logpmf instead), so every discrete fit's log_likelihood used
    # to fall back to -inf regardless of fit quality.
    registry = DistributionRegistry()
    rng = np.random.default_rng(3)
    data = stats.poisson(mu=4.0).rvs(500, random_state=rng)

    results = DistributionFitter(registry).fit(data, discrete=True)

    successes = [r for r in results if isinstance(r, FitResult)]
    assert successes, "expected at least one discrete fit to succeed"
    for fit in successes:
        assert fit.log_likelihood is not None
        assert fit.log_likelihood > float("-inf")


def test_best_likelihood_marked_even_when_no_fit_clears_the_ks_threshold():
    # _mark_best_fits used to require both a likelihood winner AND a ks winner
    # before marking either — so a best_likelihood candidate went unmarked
    # whenever every fit's ks p-value was exactly 0 (routine for discrete
    # families on a large, tied sample).
    fitter = DistributionFitter()
    worse = FitResult(
        distribution="a",
        dist_object=None,
        params=(1.0,),
        frozen_dist=None,
        log_likelihood=-100.0,
        kstest=KSTestResult(statistic=0.5, p_value=0.0),
    )
    better = FitResult(
        distribution="b",
        dist_object=None,
        params=(1.0,),
        frozen_dist=None,
        log_likelihood=-50.0,
        kstest=KSTestResult(statistic=0.5, p_value=0.0),
    )

    fitter._mark_best_fits([worse, better], criterion="combined")

    assert better.best_likelihood is True
    assert worse.best_likelihood is False
    assert better.best_ks is None or better.best_ks is False
    assert worse.best_ks is None or worse.best_ks is False


def test_mark_best_fits_rejects_unknown_criterion():
    fitter = DistributionFitter()
    with pytest.raises(ValueError, match="Unknown criterion"):
        fitter._mark_best_fits([], criterion="nonsense")


def test_fit_table_ranks_by_loglikelihood_and_sorts_failures_last():
    results = [
        FitResult(
            distribution="worse",
            dist_object=None,
            params=(1.0, 2.0),
            frozen_dist=None,
            log_likelihood=-50.0,
            kstest=KSTestResult(statistic=0.2, p_value=0.3),
            best_likelihood=False,
            best_ks=False,
        ),
        FailedFit(distribution="broken", message="optimizer did not converge"),
        FitResult(
            distribution="best",
            dist_object=None,
            params=(3.0,),
            frozen_dist=None,
            log_likelihood=-10.0,
            kstest=KSTestResult(statistic=0.05, p_value=0.9),
            best_likelihood=True,
            best_ks=True,
        ),
    ]

    table = fit_table(results)

    assert list(table["distribution"]) == ["best", "worse", "broken"]
    assert table.loc[0, "best_likelihood"]
    assert table.loc[0, "best_ks"]
    assert table.loc[2, "message"] == "optimizer did not converge"
    assert np.isnan(table.loc[2, "log_likelihood"])
