from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.category as mpc

from goad_toolkit.analytics import FitResult, Result


@dataclass
class PlotSettings:
    """Base settings for all plots."""

    figsize: Tuple[int, int] = (10, 6)
    title: str = "Plot"
    xlabel: str = "X"
    ylabel: str = "Y"
    legend_title: Optional[str] = None
    grid: bool = False  # Added grid option
    grid_alpha: Optional[float] = None  # Grid transparency
    subplot_titles: List[str] = field(default_factory=list)  # Titles for subplots

    def __repr__(self):
        return (
            f"PlotSettings(figsize={self.figsize},\n title={self.title},\n "
            f"xlabel={self.xlabel},\n ylabel={self.ylabel},\n "
            f"legend_title={self.legend_title},\n grid={self.grid},\n "
            f"grid_alpha={self.grid_alpha}"
        )


class BasePlot(ABC):
    """Base class for creating plots."""

    def __init__(self, settings: PlotSettings, n_plots: Optional[int] = None):
        self.settings = settings
        self.fig = None
        self.ax = None

    def plot(self, n_plots: Optional[int] = None, *args, **kwargs):
        """Abstract method for plotting data."""
        if self.fig is None:
            self.create_figure(n_plots)
        return self.build(*args, **kwargs)

    def create_figure(self, n_plots: Optional[int] = None):
        """Create a figure and configure it based on settings.
        Parameters:
        -----------
        n_plots : Optional[int]
            Number of subplots to create. If None, creates a single plot.
            If > 1, creates a grid of subplots.
        """
        if n_plots is None or n_plots <= 1:
            # Create a single plot
            self.fig, self.ax = plt.subplots(figsize=self.settings.figsize)
            axes = [self.ax]
        else:
            # Determine grid layout
            grid_cols = min(3, n_plots)  # Max 3 columns
            grid_rows = int(np.ceil(n_plots / grid_cols))
            # Create subplot grid
            self.fig, axes = plt.subplots(
                grid_rows, grid_cols, figsize=self.settings.figsize, squeeze=False
            )
            axes = axes.flatten()
            # Hide unused axes
            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)
            # Store first axis as default
            self.ax = axes[0]

        # Apply common settings to all axes
        for i, ax in enumerate(axes):
            if ax.get_visible():
                ax.set_xlabel(self.settings.xlabel)
                ax.set_ylabel(self.settings.ylabel)
                # Apply grid if requested
                if self.settings.grid:
                    ax.grid(True, alpha=self.settings.grid_alpha)

                # Set subplot titles if available
                if i < len(self.settings.subplot_titles):
                    ax.set_title(self.settings.subplot_titles[i])

        # Set main title
        if n_plots is None or n_plots <= 1:
            if not self.settings.subplot_titles:  # Only set if no subplot titles
                self.ax.set_title(self.settings.title)
        else:
            plt.suptitle(self.settings.title, fontsize=16)
            plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for suptitle

        if self.settings.legend_title is not None:
            self.ax.legend(title=self.settings.legend_title)

        return self.fig, axes

    def plot_on(self, other_plot: "BasePlot", *args, **kwargs):
        """Combine BasePlot classes in a hierarchy.

        Parameters:
        -----------
        other_plot : BasePlot
            The plot class to use
        *args, **kwargs : Arguments to pass to the plot method

        Returns:
        --------
        The used plot instance (after plotting)
        """
        # Create an instance of the plot class with our settings
        # other_plot = plot_class(self.settings)

        # Share our figure and axes
        other_plot.fig = self.fig
        other_plot.ax = self.ax

        # Call the plot method with the provided arguments
        other_plot.plot(*args, **kwargs)

        # Return the plot in case further configuration is needed
        return other_plot

    def plot_on_axes(self, other_plot: "BasePlot", ax: Axes, *args, **kwargs):
        """Use another plot class to plot on a specific axis within this figure.

        Parameters:
        -----------
        other_plot : BasePlot
            The plot class to use
        ax : matplotlib.axes.Axes
            The specific axes to plot on
        *args, **kwargs : Arguments to pass to the plot method

        Returns:
        --------
        The used plot instance (after plotting)
        """

        # Share our figure but use the provided axis
        other_plot.fig = self.fig
        other_plot.ax = ax

        # Call the plot method with the provided arguments
        other_plot.plot(*args, **kwargs)

        # Return the plot in case further configuration is needed
        return other_plot

    @abstractmethod
    def build(self, *args, **kwargs):
        raise NotImplementedError("Plotting method must be implemented")


class LinePlot(BasePlot):
    """Plot a line plot using seaborn."""

    def build(self, data: pd.DataFrame, **kwargs):
        sns.lineplot(data=data, ax=self.ax, **kwargs)
        return self.fig, self.ax


class ComparePlot(BasePlot):
    def build(self, data: pd.DataFrame, x: str, y1: str, y2: str, **kwargs):
        compare = LinePlot(self.settings)
        self.plot_on(compare, data=data, x=x, y=y1, label=y1, **kwargs)
        self.plot_on(compare, data=data, x=x, y=y2, label=y2, **kwargs)
        plt.xticks(rotation=45)

        return self.fig, self.ax


class VerticalDate(BasePlot):
    def build(self, date: str, label: str):
        datemark = pd.to_datetime(date)
        try:
            if not self.ax:
                raise ValueError("No axes available for plotting")
            converter = self.ax.xaxis.get_converter()
            if isinstance(converter, mpc.StrCategoryConverter):
                datemark = datemark.strftime("%Y-%m-%d")
        except AttributeError:
            logger.error("No converter found")
        plt.axvline(
            x=datemark,  # type: ignore
            color="red",
            linestyle="--",
            linewidth=2,
            label=label,
        )


class ComparePlotDate(BasePlot):
    def build(
        self,
        data: pd.DataFrame,
        x: str,
        y1: str,
        y2: str,
        date: str,
        datelabel: str,
        **kwargs,
    ):
        compare = LinePlot(self.settings)
        self.plot_on(compare, data=data, x=x, y=y1, label=y1, **kwargs)
        self.plot_on(compare, data=data, x=x, y=y2, label=y2, **kwargs)

        vertical = VerticalDate(self.settings)
        self.plot_on(vertical, date=date, label=datelabel)

        plt.xticks(rotation=45)

        return self.fig, self.ax


class BarWithDates(BasePlot):
    def build(self, data, x: str, y: str, interval: int = 1, **kwargs):
        sns.barplot(data=data, x=x, y=y, ax=self.ax, **kwargs)
        if not self.ax:
            raise ValueError("No axes available for plotting")
        self.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))


class ResidualPlot(BasePlot):
    def build(self, data, x: str, y: str, date: str, datelabel: str, interval: int = 1):
        barplot = BarWithDates(self.settings)
        self.plot_on(barplot, data=data, x=x, y=y, interval=interval)
        vertical = VerticalDate(self.settings)
        self.plot_on(vertical, date=date, label=datelabel)
        plt.xticks(rotation=45)
        return self.fig, self.ax


class HistogramPlot(BasePlot):
    """Plot a histogram using seaborn."""

    def build(
        self,
        data: np.ndarray,
        bins: Optional[int] = None,
        kde: bool = False,
        color: str = "skyblue",
        alpha: float = 0.7,
        **kwargs,
    ):
        """
        Create a histogram plot of the provided data.

        Parameters:
        -----------
        data : np.ndarray
            Data to plot
        bins : Optional[int]
            Number of bins to use
        kde : bool
            Whether to overlay a KDE plot
        color : str
            Color of the histogram
        alpha : float
            Transparency of the histogram
        **kwargs : Additional keyword arguments passed to sns.histplot

        Returns:
        --------
        fig, ax : The created figure and axes
        """
        # Calculate optimal bins if not specified
        if bins is None:
            bins = min(int(np.sqrt(len(data))), 50)  # Reasonable default

        # Plot histogram
        sns.histplot(
            data,
            bins=bins,
            kde=kde,
            color=color,
            alpha=alpha,
            ax=self.ax,
            stat="density",  # Use density for overlay compatibility
            **kwargs,
        )

        return self.fig, self.ax


class DistPlot(BasePlot):
    """Plot a parametric distribution."""

    def build(
        self,
        distribution: Any,
        x_range: Optional[Tuple[float, float]] = None,
        samples: int = 1000,
        color: str = "crimson",
        linewidth: float = 2,
        label: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Create a KDE plot from a given distribution.

        Parameters:
        -----------
        distribution : scipy.stats distribution
            Distribution to plot (must have a pdf method)
        x_range : Optional[Tuple[float, float]]
            Range of x values to plot (if None, will estimate from distribution)
        samples : int
            Number of points to sample along x-axis
        color : str
            Color of the KDE line
        linewidth : float
            Width of the KDE line
        label : Optional[str]
            Label for the plot in legend
        **kwargs : Additional keyword arguments passed to plt.plot

        Returns:
        --------
        fig, ax : The created figure and axes
        """
        x = self._estimate_x(distribution, samples, x_range)

        # Calculate probability density
        try:
            y = distribution.pdf(x)
        except AttributeError:
            # If pdf not available, try pmf for discrete distributions
            try:
                y = distribution.pmf(x)
            except (AttributeError, ValueError):
                raise ValueError("Distribution must have either pdf or pmf method")

        if self.ax is None:
            raise ValueError("No axes available for plotting")

        # Plot distribution
        self.ax.plot(x, y, color=color, linewidth=linewidth, label=label, **kwargs)

        if label is not None:
            self.ax.legend()

        return self.fig, self.ax

    def _estimate_x(
        self,
        distribution: Any,
        samples: int,
        x_range: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Estimate an appropriate x-range for plotting the distribution.
        """
        if x_range is not None:
            x = np.linspace(x_range[0], x_range[1], samples)
            return x

        # Try to use the percent point function (quantile function)
        try:
            lower = distribution.ppf(0.001)
            upper = distribution.ppf(0.999)
            # Add padding
            padding = (upper - lower) * 0.1
            # Create x values within range
            x_range = (lower - padding, upper + padding)
            x = np.linspace(x_range[0], x_range[1], samples)
            return x
        except (AttributeError, ValueError):
            logger.warning(
                "Falling back to generic range. Specify x_range for better results."
            )
            x_range = (-5, 5)
            x = np.linspace(x_range[0], x_range[1], samples)
            return x


@dataclass
class FitPlotSettings:
    """Settings for distribution fit plots."""

    bins: Optional[int] = None
    data_color: str = "lightgrey"
    best_likelihood_color: str = "crimson"
    best_ks_color: str = "darkblue"
    other_color: str = "gray"
    data_alpha: float = 0.6
    max_fits: Optional[int] = None

    def __repr__(self):
        return (
            f"FitPlotSettings(bins={self.bins},\n data_color={self.data_color},\n "
            f"best_likelihood_color={self.best_likelihood_color},\n "
            f"best_ks_color={self.best_ks_color},\n other_color={self.other_color},\n "
            f"data_alpha={self.data_alpha},\n max_fits={self.max_fits})"
        )


class PlotFits(BasePlot):
    """Plot histogram of data with fitted distributions overlaid."""

    def plot(
        self,
        data: np.ndarray,
        fit_results: List[Result],
        fitplotsettings: "FitPlotSettings",
    ) -> Figure:
        """
        Plot multiple fits on separate subplots.

        Parameters:
        -----------
        data : np.ndarray
            Data to plot
        fit_results : List[FitResult]
            List of fit results to plot (expect them to have best_likelihood and best_ks attributes)
        fitplotsettings : FitPlotSettings
            Settings for the fit plots

        Returns:
        --------
        fig : The created figure

        """
        # Filter and sort fits
        sorted_fits = self._prepare_fits(fit_results, fitplotsettings.max_fits)

        # Create subplot titles
        subplot_titles = self._create_subplot_titles(sorted_fits)
        self.settings.subplot_titles = subplot_titles

        # Create figure with subplots
        self.fig, axes = self.create_figure(n_plots=len(sorted_fits))

        # Plot each fit
        for i, fit in enumerate(sorted_fits):
            self._plot_single_fit(
                data=data, fit=fit, ax=axes[i], fitplotsettings=fitplotsettings
            )

        return self.fig

    def build(self):
        pass

    def _prepare_fits(
        self, fit_results: List[Result], max_fits: Optional[int] = None
    ) -> List[FitResult]:
        # Filter only successful fits
        successful_fits = [fit for fit in fit_results if isinstance(fit, FitResult)]

        if not successful_fits:
            raise ValueError("No successful fits to plot")

        # Sort fits by log-likelihood (descending)
        sorted_fits = sorted(
            successful_fits,
            key=lambda fit: fit.log_likelihood
            if fit.log_likelihood is not None
            else -np.inf,
            reverse=True,
        )

        # Limit number of fits if specified
        if max_fits is not None and max_fits > 0:
            sorted_fits = sorted_fits[:max_fits]

        return sorted_fits

    def _create_subplot_titles(self, fits: List[FitResult]) -> List[str]:
        subplot_titles = []
        for fit in fits:
            title = f"{fit.distribution}"
            if fit.kstest:
                title += f"\nKS p: {fit.kstest.p_value:.4f}"
            if fit.log_likelihood is not None:
                title += f", llh: {fit.log_likelihood:.2f}"
            if hasattr(fit, "best_likelihood") and fit.best_likelihood:
                title += "\n★ Best likelihood ★"
            if hasattr(fit, "best_ks") and fit.best_ks:
                title += "\n★ Best KS test ★"
            subplot_titles.append(title)
        return subplot_titles

    def _plot_single_fit(
        self, data: np.ndarray, fit: Any, ax: Axes, fitplotsettings: "FitPlotSettings"
    ) -> None:
        # Choose color based on best fit status
        dist_color = self._get_fit_color(fit, fitplotsettings)

        # Create histogram
        hist_plot = HistogramPlot(
            PlotSettings(
                xlabel=self.settings.xlabel,
                ylabel=self.settings.ylabel,
                grid=self.settings.grid,
                grid_alpha=self.settings.grid_alpha,
            )
        )
        # hist_plot.fig, hist_plot.ax = self.fig, ax
        self.plot_on_axes(
            hist_plot,
            data=data,
            ax=ax,
            bins=fitplotsettings.bins,
            kde=False,
            color=fitplotsettings.data_color,
            alpha=fitplotsettings.data_alpha,
        )

        # Overlay distribution
        kde_plot = DistPlot(PlotSettings())
        # kde_plot.fig, kde_plot.ax = self.fig, ax
        # kde_plot.plot(fit.frozen_dist, color=dist_color, label=fit.distribution)
        self.plot_on_axes(
            kde_plot,
            ax=ax,
            distribution=fit.frozen_dist,
            color=dist_color,
            label=fit.distribution,
        )

    def _get_fit_color(self, fit: Any, fitplotsettings: "FitPlotSettings") -> str:
        if hasattr(fit, "best_likelihood") and fit.best_likelihood:
            return fitplotsettings.best_likelihood_color
        elif hasattr(fit, "best_ks") and fit.best_ks:
            return fitplotsettings.best_ks_color
        else:
            return fitplotsettings.other_color
