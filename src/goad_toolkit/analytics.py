from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import optimize, stats

from goad_toolkit.distributions import DistributionRegistry


@dataclass
class KSTestResult:
    """Result of a Kolmogorov-Smirnov test."""

    statistic: float
    p_value: float
    error: Optional[str] = None


@dataclass
class FitResult:
    """Result of a successful distribution fit."""

    distribution: str  # Distribution name
    dist_object: Any  # The actual distribution object
    params: Tuple[float, ...]  # Fitted parameters
    frozen_dist: Any  # Frozen distribution with parameters applied
    success: bool = True
    message: str = "Optimization successful"
    log_likelihood: Optional[float] = None
    kstest: Optional[KSTestResult] = None
    best_likelihood: Optional[bool] = (
        None  # Indicates if this is the best fit by likelihood
    )
    best_ks: Optional[bool] = None  # Indicates if this is the best fit by KS test

    def __str__(self) -> str:
        """String representation focused on distribution and parameters."""
        return f"{self.distribution}: params={self.params}"

    def __repr__(self) -> str:
        """Detailed representation including fit quality metrics."""
        ks_info = f", ks_pvalue={self.kstest.p_value:.4f}" if self.kstest else ""
        ll_info = (
            f", loglik={self.log_likelihood:.2f}"
            if self.log_likelihood is not None
            else ""
        )
        best_info = ""
        if self.best_likelihood:
            best_info += " (Best Likelihood)"
        if self.best_ks:
            best_info += " (Best KS)"

        return f"Fit({self.distribution}: params={self.params}{ll_info}{ks_info}{best_info})"


@dataclass
class FailedFit:
    """Result of a failed distribution fit."""

    distribution: str
    message: str
    success: bool = False

    def __str__(self) -> str:
        """String representation with error message."""
        return f"Failed {self.distribution}: {self.message}"


# Type alias for either success or failure
Result = Union[FitResult, FailedFit]


class DistributionFitter:
    """Class to fit distributions from a registry to data."""

    def __init__(
        self, registry: Optional[DistributionRegistry] = None, seed: Optional[int] = 42
    ) -> None:
        """Initialize with a distribution registry.

        Pass a registry to fit a custom set of families; a fresh registry with
        the shipped families is used otherwise.

        `seed` fixes the optimizer: `stats.fit` searches the bounds with
        differential evolution, which is stochastic, so without a seed the same
        data can come back with slightly different parameters on every run.
        Pass `None` to let scipy draw its own.
        """
        self._registry = registry if registry is not None else DistributionRegistry()
        self.seed = seed
        self._optimizer = (
            optimize.differential_evolution
            if seed is None
            else partial(optimize.differential_evolution, seed=seed)
        )

    @property
    def registry(self) -> list[str]:
        return self._registry.get_names()

    def _get_bounds(self, data: np.ndarray, dist_obj) -> list[tuple]:
        """Generate parameter bounds estimates based on data characteristics."""
        # Calculate statistics for bounds
        data_min = np.min(data)
        data_max = np.max(data)
        data_mean = np.mean(data)
        data_std = np.std(data)
        data_range = data_max - data_min

        # Universal parameter bounds
        loc_bound = (data_min - data_range, data_max + data_range)
        std_lower_bound = float(data_std / 100)
        scale_bound = (max(std_lower_bound, 0.001), data_std * 20)
        shape_bound = (0.01, 10.0)

        # Create bounds based on parameter count
        if dist_obj.num_params == 1:
            if dist_obj.is_discrete:
                # A single-parameter discrete family's shape parameter is not always
                # a count-scale rate (poisson's lambda) — it can be a probability
                # (bernoulli's p), which is routinely well under 0.1, so the floor
                # here stays near zero rather than at 0.1.
                mean_lower_bound = max(1e-3, float(data_mean / 10))
                mean_upper_bound = float(data_mean * 10)
                return [(mean_lower_bound, mean_upper_bound)]
            else:
                return [scale_bound]
        elif dist_obj.num_params == 2:
            if dist_obj.is_discrete:
                # Two-parameter discrete families (binomial, nbinom) take a count
                # `n` and a probability `p`, not a location and a scale. `n` is a
                # trial count for binomial and a dispersion for nbinom, so it gets
                # a count-scale range; `p` has to stay inside (0, 1).
                count_bound = (0.01, max(10_000.0, float(data_max) * 10))
                probability_bound = (1e-6, 1.0)
                return [count_bound, probability_bound]
            return [loc_bound, scale_bound]
        elif dist_obj.num_params == 3:
            return [shape_bound, loc_bound, scale_bound]
        else:
            return [shape_bound] * (dist_obj.num_params - 2) + [loc_bound, scale_bound]

    def _perform_kstest(self, data: np.ndarray, dist_obj, params) -> KSTestResult:
        """Perform Kolmogorov-Smirnov test for goodness-of-fit."""
        try:
            # Create a frozen distribution with fitted parameters
            fitted_dist = dist_obj.dist(*params)

            # Run KS test - comparing data with the fitted distribution
            ks_statistic, p_value = stats.kstest(data, fitted_dist.cdf)

            return KSTestResult(statistic=ks_statistic, p_value=p_value)
        except Exception as e:
            return KSTestResult(
                statistic=float("nan"), p_value=float("nan"), error=str(e)
            )

    def _calculate_loglikelihood(self, data: np.ndarray, dist_obj, params) -> float:
        """Calculate log-likelihood of data given distribution and parameters.
        For every datapoint, calculate the log density (continuous families, via
        logpdf) or the log mass (discrete families, via logpmf). We sum them to get
        the log-likelihood of the data. If there are a lot of datapoints with very
        low probability, the log-likelihood will be very negative (or -inf if
        probability is 0).

        We will prefer distributions with higher log-likelihood values.
        """
        try:
            log_density = (
                dist_obj.dist.logpmf if dist_obj.is_discrete else dist_obj.dist.logpdf
            )
            return np.sum(log_density(data, *params))
        except Exception as e:
            logger.warning(f"Log-likelihood calculation failed: {str(e)}")
            return -np.inf

    def fit_distribution(
        self, dist_name: str, data: np.ndarray, method: str = "mle"
    ) -> Result:
        """
        Fit a specific distribution to data.

        Args:
            dist_name: Name of the distribution to fit
            data: Data to fit the distribution to
            method: Fitting method (default: 'mle')

        Returns:
            FitResult or FailedFit
        """
        try:
            dist_obj = self._registry.get_distribution(dist_name)

            if dist_obj.is_discrete:
                # A discrete family's support does not depend on its parameters
                # (bernoulli lives on {0, 1}; poisson, binomial and nbinom on the
                # non-negative integers), so data outside it is a mismatch to
                # report, not an optimisation to attempt.
                low, high = dist_obj.dist.a, dist_obj.dist.b
                if np.min(data) < low or np.max(data) > high:
                    message = f"values outside the family's support [{low}, {high}]"
                    return FailedFit(distribution=dist_name, message=message)

            # Get parameter bounds
            bounds = self._get_bounds(data, dist_obj)
            # logger.info(f"Bounds for {dist_name}: {bounds}")

            # Perform the fit
            result = stats.fit(
                dist_obj.dist,
                data,
                method=method,
                bounds=tuple(bounds),
                optimizer=self._optimizer,
            )

            # If fit was not successful, return failure
            if not result.success:
                logger.warning(f"Fitting failed: {result.message}")
                return FailedFit(distribution=dist_name, message=str(result.message))

            # Create frozen distribution with fitted parameters
            frozen_dist = dist_obj.dist(*result.params)

            # Run goodness-of-fit tests
            kstest_result = self._perform_kstest(data, dist_obj, result.params)
            log_likelihood = self._calculate_loglikelihood(
                data, dist_obj, result.params
            )

            return FitResult(
                distribution=dist_name,
                dist_object=dist_obj.dist,
                params=result.params,
                frozen_dist=frozen_dist,
                message=str(result.message),
                log_likelihood=log_likelihood,
                kstest=kstest_result,
            )

        except Exception as e:
            logger.warning(f"Fitting failed: {str(e)}")
            return FailedFit(
                distribution=dist_name, message=f"Fitting failed: {str(e)}"
            )

    def _mark_best_fits(self, fits: List[Result], criterion: str) -> List[Result]:
        """
        Mark the best fits in a list of fit results.

        Args:
            fits: List of fit results to analyze
            criterion: Selection criterion ('likelihood', 'ks', or 'combined')

        Returns:
            The same list with best_likelihood and best_ks attributes updated for successful fits
        """
        # Find best likelihood fit
        best_likelihood_value = float("-inf")
        best_likelihood_fit = None

        # Find best KS test fit
        best_ks_value = 0
        best_ks_fit = None

        # Find the best fits without creating a filtered list
        for fit in fits:
            if isinstance(fit, FitResult):
                # Check for best likelihood
                fit_likelihood = (
                    fit.log_likelihood
                    if fit.log_likelihood is not None
                    else float("-inf")
                )
                if fit_likelihood > best_likelihood_value:
                    best_likelihood_value = fit_likelihood
                    best_likelihood_fit = fit

                # Check for best KS test
                fit_ks = fit.kstest.p_value if fit.kstest else 0
                if fit_ks > best_ks_value:
                    best_ks_value = fit_ks
                    best_ks_fit = fit

        if criterion not in ("likelihood", "ks", "combined"):
            raise ValueError(f"Unknown criterion '{criterion}'")

        # Each criterion is marked independently: a KS test that never clears the
        # p > 0 threshold (routine for discrete families on a large sample, where
        # ties push the p-value to numerical zero) must not suppress a perfectly
        # good best_likelihood winner, and vice versa.
        if criterion in ("likelihood", "combined") and best_likelihood_fit is not None:
            for fit in fits:
                if isinstance(fit, FitResult):
                    fit.best_likelihood = (
                        fit.distribution == best_likelihood_fit.distribution
                    )

        if criterion in ("ks", "combined") and best_ks_fit is not None:
            for fit in fits:
                if isinstance(fit, FitResult):
                    fit.best_ks = fit.distribution == best_ks_fit.distribution

        # Return the original list
        return fits

    def fit(
        self,
        data: np.ndarray,
        discrete: bool,
        method: str = "mle",
        criterion: str = "combined",
    ) -> list[Result]:
        """
        Fit all registered distributions to data, mark the best fits, and return results.

        Args:
            data: Data to fit the distribution to
            discrete: Whether to fit discrete (True) or continuous (False) distributions
            method: Fitting method (default: 'mle')
            criterion: Selection criterion ('likelihood', 'ks', or 'combined') for marking best fits

        Returns:
            List of Result objects (either FitResult or FailedFit) with best fits marked
        """
        if criterion not in ["likelihood", "ks", "combined"]:
            raise ValueError(f"Unknown criterion '{criterion}'")

        results = []
        for dist_name in self._registry.get_names():
            dist_obj = self._registry.get_distribution(dist_name)

            if discrete and dist_obj.is_discrete:
                results.append(self.fit_distribution(dist_name, data, method))

            if not discrete and not dist_obj.is_discrete:
                results.append(self.fit_distribution(dist_name, data, method))

        # Mark the best fits based on criterion
        self._mark_best_fits(results, criterion)

        return results

    @staticmethod
    def best(results: list[Result], criterion: str = "combined") -> list[Result]:
        if criterion == "ks":
            return [
                fit for fit in results if isinstance(fit, FitResult) and fit.best_ks
            ]
        elif criterion == "likelihood":
            return [
                fit
                for fit in results
                if isinstance(fit, FitResult) and fit.best_likelihood
            ]
        elif criterion == "combined":
            return [
                fit
                for fit in results
                if isinstance(fit, FitResult) and (fit.best_likelihood or fit.best_ks)
            ]
        else:
            raise ValueError(f"Unknown criterion '{criterion}'")


@dataclass
class NullResult:
    """A statistic, and the same statistic measured under shuffled labels.

    `observed` is the number you found. `null` holds the same measurement taken
    `n_iter` times on data where the labels were shuffled — data in which, by
    construction, the label explains nothing. If `observed` sits inside that
    cloud, a world where nothing is happening produces your finding routinely.
    """

    observed: float
    null: np.ndarray
    n_iter: int
    seed: Optional[int] = None

    def p_value(self, alternative: str = "two-sided") -> float:
        """How often a shuffle matched or beat the observed value.

        Args:
            alternative: `"greater"` counts shuffles at or above `observed`,
                `"less"` counts shuffles at or below it, and `"two-sided"`
                (the default) is twice the smaller of the two, capped at 1.

        Both one-sided counts include the observed arrangement itself — it is
        one of the arrangements the labels could take — so the smallest value
        this can return is `1 / (n_iter + 1)` rather than zero. With 1000
        shuffles, "p < 0.001" is the strongest statement available; report it
        that way rather than as an exact number.
        """
        greater = (int(np.sum(self.null >= self.observed)) + 1) / (self.n_iter + 1)
        less = (int(np.sum(self.null <= self.observed)) + 1) / (self.n_iter + 1)

        if alternative == "greater":
            return greater
        if alternative == "less":
            return less
        if alternative == "two-sided":
            return min(1.0, 2 * min(greater, less))
        raise ValueError(
            f"Unknown alternative '{alternative}', "
            "expected 'two-sided', 'greater' or 'less'"
        )

    @property
    def quantile(self) -> float:
        """Where `observed` sits in the null, as a fraction between 0 and 1."""
        return float(np.mean(self.null < self.observed))

    def __repr__(self) -> str:
        return (
            f"NullResult(observed={self.observed:.4g}, "
            f"null_mean={float(np.mean(self.null)):.4g}, "
            f"quantile={self.quantile:.3f}, "
            f"p={self.p_value():.4f}, n_iter={self.n_iter})"
        )


class NullDistribution:
    """Shuffle the labels, measure again, and see where the real number lands.

    The cheapest honest check in the method (§9.6): it assumes nothing about the
    shape of the data, needs no table of critical values, and answers in a
    picture rather than a test. Pair it with `NullPlot` — a student who has seen
    their finding sitting in the middle of the null does not need a p-value
    explained.

    The statistic is yours to write. Naming what you are measuring is the
    thinking part; running it a thousand times is the part worth delegating.

    ```python
    def gap(frame):
        means = frame.groupby("is_bot")["length"].mean()
        return means[True] - means[False]

    result = NullDistribution(gap, n_iter=1000, seed=42).run(data, label="is_bot")
    ```
    """

    def __init__(
        self,
        statistic: Callable[[pd.DataFrame], float],
        n_iter: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            statistic: Takes a dataframe, returns one number. Called once on the
                data as given, then once per shuffle.
            n_iter: Number of shuffles. 1000 is enough to see the shape; go to
                10_000 when the observed value sits far out in the tail.
            seed: Passed to `numpy.random.default_rng`. Set it, or the picture
                changes every time the cell is run.
        """
        if n_iter < 1:
            raise ValueError(f"n_iter must be at least 1, got {n_iter}")
        self.statistic = statistic
        self.n_iter = n_iter
        self.seed = seed

    def run(self, data: pd.DataFrame, label: str) -> NullResult:
        """Measure `statistic` on `data`, then on `n_iter` shuffles of `label`.

        Shuffling one column breaks the link between it and every other column
        while leaving both distributions exactly as they were: the labels keep
        their proportions, the values keep their shape, and only the pairing is
        destroyed. That is the "nothing is happening" world to compare against.

        Args:
            data: The frame to measure. It is never modified.
            label: Column to shuffle — the one whose association with the rest
                of the frame is in question.

        Returns:
            A `NullResult` holding the observed value and the shuffled cloud.
        """
        if label not in data.columns:
            raise ValueError(
                f"Column '{label}' is not in the data; got {list(data.columns)}"
            )

        observed = float(self.statistic(data))
        if not np.isfinite(observed):
            raise ValueError(
                f"The statistic returned {observed} on the unshuffled data; "
                "there is nothing to compare a null against."
            )

        rng = np.random.default_rng(self.seed)
        values = data[label].to_numpy()
        shuffled = data.copy()

        null = np.empty(self.n_iter)
        for i in range(self.n_iter):
            # A plain array, not a Series: assigning a permuted Series realigns
            # on the index and puts every label straight back where it started.
            shuffled[label] = rng.permutation(values)
            null[i] = float(self.statistic(shuffled))

        if not np.all(np.isfinite(null)):
            logger.warning(
                f"{int(np.sum(~np.isfinite(null)))} of {self.n_iter} shuffles gave a "
                "non-finite statistic; a small group may be emptying under shuffling."
            )

        return NullResult(
            observed=observed, null=null, n_iter=self.n_iter, seed=self.seed
        )


def fit_table(results: List[Result]) -> pd.DataFrame:
    """Turn a list of fit results into a ranked dataframe: what a student hands in.

    One row per distribution: name, fitted params, log-likelihood, KS statistic and
    KS p-value, plus which criteria (if any) it won. Ranked by log-likelihood,
    descending; `FailedFit` entries carry their failure message in `message` and
    sort last, since a missing likelihood is not a small likelihood.
    """
    rows = []
    for result in results:
        if isinstance(result, FitResult):
            rows.append(
                {
                    "distribution": result.distribution,
                    "params": result.params,
                    "log_likelihood": result.log_likelihood,
                    "ks_stat": result.kstest.statistic
                    if result.kstest
                    else float("nan"),
                    "ks_pvalue": result.kstest.p_value
                    if result.kstest
                    else float("nan"),
                    "best_likelihood": bool(result.best_likelihood),
                    "best_ks": bool(result.best_ks),
                    "message": result.message,
                }
            )
        else:
            rows.append(
                {
                    "distribution": result.distribution,
                    "params": None,
                    "log_likelihood": float("nan"),
                    "ks_stat": float("nan"),
                    "ks_pvalue": float("nan"),
                    "best_likelihood": False,
                    "best_ks": False,
                    "message": result.message,
                }
            )

    table = pd.DataFrame(rows)
    return table.sort_values(
        "log_likelihood", ascending=False, na_position="last"
    ).reset_index(drop=True)
