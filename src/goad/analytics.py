from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from scipy import stats

from goad.distributions import DistributionRegistry


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

    def __init__(self) -> None:
        """Initialize with a distribution registry."""
        self._registry = DistributionRegistry()

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
                mean_lower_bound = max(0.1, float(data_mean / 10))
                mean_upper_bound = float(data_mean * 10)
                return [(mean_lower_bound, mean_upper_bound)]
            else:
                return [scale_bound]
        elif dist_obj.num_params == 2:
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
        For every datapoint, calculate the log probability density function (PDF)
        We sum all the log PDFs to get the log-likelihood of the data.
        If there are a lot of datapoints with very low probability, the log-likelihood
        will be very negative (or -inf if probability is 0).

        We will prefer distributions with higher log-likelihood values.
        """
        try:
            return np.sum(dist_obj.dist.logpdf(data, *params))
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

            # Get parameter bounds
            bounds = self._get_bounds(data, dist_obj)
            # logger.info(f"Bounds for {dist_name}: {bounds}")

            # Perform the fit
            result = stats.fit(dist_obj.dist, data, method=method, bounds=tuple(bounds))

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

        # Only mark if we found best fits
        if best_likelihood_fit is not None and best_ks_fit is not None:
            # Mark based on criterion
            if criterion == "likelihood":
                # Mark only likelihood best
                for fit in fits:
                    if isinstance(fit, FitResult):
                        fit.best_likelihood = (
                            fit.distribution == best_likelihood_fit.distribution
                        )

            elif criterion == "ks":
                # Mark only KS best
                for fit in fits:
                    if isinstance(fit, FitResult):
                        fit.best_ks = fit.distribution == best_ks_fit.distribution

            elif criterion == "combined":
                # Mark both
                for fit in fits:
                    if isinstance(fit, FitResult):
                        fit.best_likelihood = (
                            fit.distribution == best_likelihood_fit.distribution
                        )
                        fit.best_ks = fit.distribution == best_ks_fit.distribution
            else:
                raise ValueError(f"Unknown criterion '{criterion}'")

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
