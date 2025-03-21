from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any
from goad.distributions import DistributionRegistry
import numpy as np
from scipy import stats
from loguru import logger


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
        return f"Fit({self.distribution}: params={self.params}{ll_info}{ks_info})"


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
        self.registry = DistributionRegistry()

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
            dist_obj = self.registry.get_distribution(dist_name)

            # Get parameter bounds
            bounds = self._get_bounds(data, dist_obj)

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

    def fit(
        self, data: np.ndarray, discrete: bool, method: str = "mle"
    ) -> list[Result]:
        """
        Fit all registered distributions to data and return results.

        Args:
            data: Data to fit the distribution to
            method: Fitting method (default: 'mle')
            fit_discrete_distributions: Whether to fit discrete distributions (default: True)

        Returns:
            List of Result objects (either FitResult or FailedFit)
        """
        results = []
        for dist_name in self.registry.get_names():
            dist_obj = self.registry.get_distribution(dist_name)

            if discrete and dist_obj.is_discrete:
                results.append(self.fit_distribution(dist_name, data, method))

            if not discrete and not dist_obj.is_discrete:
                results.append(self.fit_distribution(dist_name, data, method))
        return results

    def best(
        self,
        data: np.ndarray,
        discrete: bool,
        criterion: str,
        method: str = "mle",
    ) -> FitResult | tuple[FitResult, FitResult]:
        """
        Find the best fitting distribution.

        Args:
            data: Data to fit distributions to
            method: Fitting method (default: 'mle')
            criterion: Selection criterion ('likelihood', 'ks', or 'combined')

        Returns:
            Best SuccessfulFitResult or None if no successful fits
        """
        fits = self.fit(data=data, discrete=discrete, method=method)
        succesful_fits = [fit for fit in fits if isinstance(fit, FitResult)]

        # Select the best fit based on chosen criterion
        if criterion == "likelihood":
            # Use log-likelihood only
            return max(succesful_fits, key=lambda fit: fit.log_likelihood or -np.inf)
        elif criterion == "ks":
            # Use KS p-value only (higher is better)
            return max(
                succesful_fits, key=lambda fit: fit.kstest.p_value if fit.kstest else 0
            )
        elif criterion == "combined":
            likelihood = max(
                succesful_fits, key=lambda fit: fit.log_likelihood or -np.inf
            )
            ks = max(
                succesful_fits, key=lambda fit: fit.kstest.p_value if fit.kstest else 0
            )
            if likelihood.distribution == ks.distribution:
                return likelihood
            else:
                return (likelihood, ks)
        else:
            raise ValueError(f"Unknown criterion '{criterion}'")
