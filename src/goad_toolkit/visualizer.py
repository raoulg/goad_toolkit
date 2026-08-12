from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import matplotlib.category as mpc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from goad_toolkit.analytics import FitResult, NullResult, Result


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
    highlight: List[str] = field(default_factory=list)  # Categories to pick out
    base_color: str = "lightgrey"  # Everything that is not highlighted
    highlight_color: str = "crimson"  # The categories that are

    def __repr__(self):
        return (
            f"PlotSettings(figsize={self.figsize},\n title={self.title},\n "
            f"xlabel={self.xlabel},\n ylabel={self.ylabel},\n "
            f"legend_title={self.legend_title},\n grid={self.grid},\n "
            f"grid_alpha={self.grid_alpha},\n highlight={self.highlight},\n "
            f"base_color={self.base_color},\n highlight_color={self.highlight_color}"
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
            self.fig.suptitle(self.settings.title, fontsize=16)
            self.fig.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for suptitle

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
        if self.ax is None:
            raise ValueError("No axes available for plotting")
        self.ax.tick_params(axis="x", rotation=45)

        return self.fig, self.ax


class VerticalDate(BasePlot):
    def build(self, date: str, label: str):
        if self.ax is None:
            raise ValueError("No axes available for plotting")
        datemark = pd.to_datetime(date)
        try:
            converter = self.ax.xaxis.get_converter()
            if isinstance(converter, mpc.StrCategoryConverter):
                datemark = datemark.strftime("%Y-%m-%d")
        except AttributeError:
            logger.error("No converter found")
        self.ax.axvline(
            x=datemark,  # type: ignore
            color="red",
            linestyle="--",
            linewidth=2,
            label=label,
        )
        return self.fig, self.ax


class VerticalLine(BasePlot):
    """Mark one value on the x-axis. `VerticalDate` is the same idea for dates."""

    def build(
        self,
        x: float,
        label: Optional[str] = None,
        color: str = "crimson",
        linestyle: str = "--",
        linewidth: float = 2,
        **kwargs,
    ):
        """
        Draw a vertical line at `x`.

        Parameters:
        -----------
        x : float
            Position on the x-axis.
        label : Optional[str]
            Legend label. The line is drawn without a legend when this is None.
        color : str
            Line colour.
        linestyle : str
            Line style.
        linewidth : float
            Line width.
        **kwargs : Additional keyword arguments passed to `ax.axvline`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        if self.ax is None:
            raise ValueError("No axes available for plotting")

        self.ax.axvline(
            x=x,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            **kwargs,
        )
        if label is not None:
            self.ax.legend()

        return self.fig, self.ax


class HighlightCategory(BasePlot):
    """Grey everything, then colour the categories the plot is actually about.

    A layer rather than a setting, so it needs no cooperation from whatever drew
    the bars: it recolours the artists already on the axis. That works over a
    `BasePlot` subclass you wrote yourself, over `sns.barplot`, and over pandas'
    own `.plot.bar` — none of which have to know this class exists.

    Colour is a pointer, not decoration. Twenty-one bars in twenty-one colours
    say "look at all of these"; twenty grey bars and one red one say what the
    plot is for.
    """

    def build(
        self,
        categories: Optional[List[str]] = None,
        color: Optional[str] = None,
        base_color: Optional[str] = None,
        axis: str = "x",
    ):
        """
        Recolour the bars on this axis: `categories` picked out, the rest grey.

        Parameters:
        -----------
        categories : Optional[List[str]]
            Category labels to highlight. Falls back to `settings.highlight`.
        color : Optional[str]
            Colour for the highlighted bars. Falls back to
            `settings.highlight_color`.
        base_color : Optional[str]
            Colour for everything else. Falls back to `settings.base_color`.
        axis : str
            `"x"` for vertical bars (the categories are on the x-axis), `"y"`
            for horizontal ones.

        A grouped (`hue`) chart has several bars per category, and all of them
        are recoloured together. Its legend is left as seaborn drew it, and so
        stops describing the bars — drop the legend when you highlight, since
        the highlight is now the message and the hue is not.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        if self.ax is None:
            raise ValueError("No axes available for plotting")
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")

        wanted = (
            list(categories)
            if categories is not None
            else list(self.settings.highlight)
        )
        if not wanted:
            raise ValueError(
                "Nothing to highlight: pass categories=[...] or set "
                "PlotSettings(highlight=[...])"
            )

        # Bars are Rectangles, and seaborn parks zero-size ones in ax.patches as
        # legend proxies: they carry no data, and recolouring them would rewrite
        # the legend's swatches.
        bars = [
            patch
            for patch in self.ax.patches
            if isinstance(patch, Rectangle)
            and (patch.get_width() != 0 or patch.get_height() != 0)
        ]
        if not bars:
            raise ValueError(
                "No bars on this axis to recolour. HighlightCategory is a layer — "
                "draw the chart first, then apply it with plot_on."
            )

        if axis == "x":
            ticks = self.ax.get_xticks()
            labels = [text.get_text() for text in self.ax.get_xticklabels()]
        else:
            ticks = self.ax.get_yticks()
            labels = [text.get_text() for text in self.ax.get_yticklabels()]

        unknown = [category for category in wanted if category not in labels]
        if unknown:
            raise ValueError(
                f"No category {unknown} on the {axis}-axis; it has {labels}"
            )

        color = color if color is not None else self.settings.highlight_color
        base_color = base_color if base_color is not None else self.settings.base_color

        for patch in bars:
            if axis == "x":
                centre = patch.get_x() + patch.get_width() / 2
            else:
                centre = patch.get_y() + patch.get_height() / 2
            # Bars sit at their category's tick; the nearest one names them,
            # which holds for grouped bars offset around it as well.
            nearest = int(np.argmin(np.abs(np.asarray(ticks) - centre)))
            label = labels[nearest] if nearest < len(labels) else ""
            patch.set_facecolor(color if label in wanted else base_color)

        return self.fig, self.ax


class Annotate(BasePlot):
    """Write the finding on the plot, where a reader with thirty seconds looks.

    The last question in the visual critique is whether the finding is written
    down. Axis labels say what the numbers are; the annotation says what you
    concluded, and it is the only part most readers will read.
    """

    def build(
        self,
        text: str,
        xy: Tuple[float, float],
        xytext: Optional[Tuple[float, float]] = None,
        arrow: bool = True,
        color: str = "black",
        fontsize: float = 11,
        **kwargs,
    ):
        """
        Place `text` on the axis, optionally with an arrow to the point.

        Parameters:
        -----------
        text : str
            What you concluded. A sentence, not a label.
        xy : Tuple[float, float]
            Point in data coordinates the text is about.
        xytext : Optional[Tuple[float, float]]
            Where the text itself sits, in data coordinates. Defaults to `xy`,
            which places the text on the point and draws no arrow.
        arrow : bool
            Draw an arrow from the text to `xy`. Ignored when `xytext` is None,
            since an arrow from a point to itself is not a thing.
        color : str
            Text colour.
        fontsize : float
            Text size.
        **kwargs : Additional keyword arguments passed to `ax.annotate`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        if self.ax is None:
            raise ValueError("No axes available for plotting")

        arrowprops = None
        if xytext is not None and arrow:
            arrowprops = {"arrowstyle": "->", "color": color, "linewidth": 1}

        self.ax.annotate(
            text,
            xy=xy,
            xytext=xytext if xytext is not None else xy,
            color=color,
            fontsize=fontsize,
            arrowprops=arrowprops,
            **kwargs,
        )
        return self.fig, self.ax


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

        if self.ax is None:
            raise ValueError("No axes available for plotting")
        self.ax.tick_params(axis="x", rotation=45)

        return self.fig, self.ax


class BarPlot(BasePlot):
    """One bar per category. The plot for "how do these groups differ".

    Ordering and the top-k cut are data decisions, so they belong in a
    `Pipeline` rather than here — pass the frame already sorted, or hand
    `order` through to seaborn. Alphabetical is almost never meaningful order,
    and past a dozen bars nobody is comparing anything.

    Colour is left to seaborn: one bar, one colour, and `HighlightCategory` on
    top when a single category is the point.
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        order: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Draw one bar per level of `x`.

        Parameters:
        -----------
        data : pd.DataFrame
            Frame holding both columns.
        x : str
            Categorical column, one bar per level.
        y : str
            Numeric column. Seaborn aggregates it per category with the mean
            unless you pass another `estimator`.
        order : Optional[List[str]]
            Category order. Seaborn's own order (first appearance) is used
            when this is None.
        **kwargs : Additional keyword arguments passed to `sns.barplot`,
            including `estimator`, `errorbar`, `color` and `hue`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        sns.barplot(data=data, x=x, y=y, order=order, ax=self.ax, **kwargs)
        return self.fig, self.ax


class GroupedBarPlot(BasePlot):
    """Two categorical variables at once: bars per group, split by `hue`.

    Here colour carries data rather than emphasis, which is the one case where
    a full palette is right. Past two or three hue levels the groups stop being
    comparable — that is when the question wants `HeatmapPlot` instead.
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: str,
        order: Optional[List[str]] = None,
        hue_order: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Draw bars of `y` per level of `x`, split by `hue`.

        Parameters:
        -----------
        data : pd.DataFrame
            Frame holding all three columns.
        x, y : str
            Category and value columns.
        hue : str
            Second categorical column. Required: without it this is `BarPlot`.
        order, hue_order : Optional[List[str]]
            Category and hue orders.
        **kwargs : Additional keyword arguments passed to `sns.barplot`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        sns.barplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            ax=self.ax,
            **kwargs,
        )
        return self.fig, self.ax


class HeatmapPlot(BasePlot):
    """Two categorical variables with many levels each, as a grid.

    Where a grouped bar chart runs out: twelve groups by seven hue levels is
    unreadable as bars and legible as a grid. The scale is sequential by
    default, because a count or a mean has a floor and no meaningful centre —
    `CorrelationHeatmap` is the diverging case.
    """

    def build(
        self,
        data: pd.DataFrame,
        index: Optional[str] = None,
        columns: Optional[str] = None,
        values: Optional[str] = None,
        aggfunc: str = "mean",
        cmap: str = "rocket_r",
        annot: bool = True,
        fmt: str = ".1f",
        **kwargs,
    ):
        """
        Draw a heatmap, pivoting the frame first when told how.

        Parameters:
        -----------
        data : pd.DataFrame
            Long frame to pivot, or a matrix already shaped rows × columns.
        index, columns : Optional[str]
            Categorical columns for the rows and columns of the grid. Both
            must be given to pivot; leave both out to plot `data` as it is.
        values : Optional[str]
            Column to aggregate. With `aggfunc="count"` any column will do.
        aggfunc : str
            Passed to `DataFrame.pivot_table`.
        cmap : str
            Colormap. Sequential by default, dark for high.
        annot : bool
            Write each cell's value into it.
        fmt : str
            Format for the annotations.
        **kwargs : Additional keyword arguments passed to `sns.heatmap`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        if (index is None) != (columns is None):
            raise ValueError(
                "Pass both index and columns to pivot, or neither to plot the "
                "frame as it is"
            )

        if index is not None and columns is not None:
            if values is None:
                raise ValueError("Pivoting needs a values column to aggregate")
            matrix = data.pivot_table(
                index=index, columns=columns, values=values, aggfunc=aggfunc
            )
        else:
            matrix = data

        sns.heatmap(matrix, cmap=cmap, annot=annot, fmt=fmt, ax=self.ax, **kwargs)
        return self.fig, self.ax


class BarbellPlot(BasePlot):
    """Before and after per category, with the gap between them drawn.

    Two bar charts side by side make the reader subtract; a barbell draws the
    subtraction. Use it when the *change* is the finding rather than either
    level.
    """

    def build(
        self,
        data: pd.DataFrame,
        category: str,
        start: str,
        end: str,
        start_label: str = "before",
        end_label: str = "after",
        start_color: str = "lightgrey",
        end_color: str = "crimson",
        line_color: str = "lightgrey",
        markersize: float = 80,
        **kwargs,
    ):
        """
        Draw one row per category, with a dot at `start`, a dot at `end`, and
        a line joining them.

        Parameters:
        -----------
        data : pd.DataFrame
            Frame with one row per category. Rows are drawn in the order given
            — sorting by the size of the change is usually what you want, and
            it belongs upstream of the plot.
        category : str
            Column naming each row; becomes the y-axis labels.
        start, end : str
            Columns holding the two values to join.
        start_label, end_label : str
            Legend labels for the two dots.
        start_color, end_color, line_color : str
            Colours for the two dots and the line between them.
        markersize : float
            Dot size, passed to `ax.scatter` as `s`.
        **kwargs : Additional keyword arguments passed to `ax.scatter`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        if self.ax is None:
            raise ValueError("No axes available for plotting")

        missing = [c for c in (category, start, end) if c not in data.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} not in the data")

        positions = np.arange(len(data))
        starts = data[start].to_numpy()
        ends = data[end].to_numpy()

        self.ax.hlines(
            y=positions, xmin=starts, xmax=ends, color=line_color, linewidth=2, zorder=1
        )
        self.ax.scatter(
            starts,
            positions,
            s=markersize,
            color=start_color,
            label=start_label,
            zorder=2,
            **kwargs,
        )
        self.ax.scatter(
            ends,
            positions,
            s=markersize,
            color=end_color,
            label=end_label,
            zorder=2,
            **kwargs,
        )
        self.ax.set_yticks(positions)
        self.ax.set_yticklabels([str(label) for label in data[category]])
        self.ax.legend()

        return self.fig, self.ax


class BarWithDates(BasePlot):
    def build(self, data, x: str, y: str, interval: int = 1, **kwargs):
        if self.ax is None:
            raise ValueError("No axes available for plotting")
        sns.barplot(data=data, x=x, y=y, ax=self.ax, **kwargs)
        self.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        return self.fig, self.ax


class ResidualPlot(BasePlot):
    def build(self, data, x: str, y: str, date: str, datelabel: str, interval: int = 1):
        barplot = BarWithDates(self.settings)
        self.plot_on(barplot, data=data, x=x, y=y, interval=interval)
        vertical = VerticalDate(self.settings)
        self.plot_on(vertical, date=date, label=datelabel)
        if self.ax is None:
            raise ValueError("No axes available for plotting")
        self.ax.tick_params(axis="x", rotation=45)
        return self.fig, self.ax


class DecomposePlot(BasePlot):
    """Decompose a series into observed, trend, seasonal and residual panels."""

    def plot(
        self,
        data: pd.DataFrame,
        column: str,
        period: int,
        model: str = "additive",
        **kwargs,
    ) -> Figure:
        """
        Run `statsmodels.tsa.seasonal.seasonal_decompose` and draw each
        component on its own panel.

        Parameters:
        -----------
        data : pd.DataFrame
            Frame with a datetime index and the series to decompose.
        column : str
            Column to decompose.
        period : int
            Number of observations per cycle (e.g. 24 for hourly data with a
            daily cycle, 7 for daily data with a weekly cycle).
        model : str
            `"additive"` or `"multiplicative"`, passed to `seasonal_decompose`.
        **kwargs : Additional keyword arguments passed to `seasonal_decompose`.

        Returns:
        --------
        fig : The created figure, with one panel per component.
        """
        result = seasonal_decompose(data[column], model=model, period=period, **kwargs)
        components = [
            ("Observed", result.observed),
            ("Trend", result.trend),
            ("Seasonal", result.seasonal),
            ("Residual", result.resid),
        ]

        self.settings.subplot_titles = [name for name, _ in components]
        self.fig, axes = self.create_figure(n_plots=len(components))

        for ax, (_, values) in zip(axes, components):
            comp_data = values.rename("value").reset_index()
            index_col = comp_data.columns[0]
            line = LinePlot(self.settings)
            self.plot_on_axes(line, ax=ax, data=comp_data, x=index_col, y="value")

        return self.fig

    def build(self):
        """No-op: `DecomposePlot` overrides `plot` because the panel count is fixed at four."""
        pass


class ACFPlot(BasePlot):
    """Bar plot of the autocorrelation function with a Bartlett confidence band."""

    def build(
        self,
        data: Any,
        nlags: int = 40,
        alpha: float = 0.05,
        color: str = "steelblue",
        band_color: str = "gray",
        **kwargs,
    ):
        """
        Plot the autocorrelation function for a series.

        Parameters:
        -----------
        data : array-like
            Series to compute the ACF for. Must not contain missing values —
            drop NaNs first (a decomposition's residual has them at the edges).
        nlags : int
            Number of lags to compute and plot.
        alpha : float
            Significance level for the confidence band.
        color : str
            Bar color.
        band_color : str
            Confidence band color.
        **kwargs : Additional keyword arguments passed to `ax.bar`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        if self.ax is None:
            raise ValueError("No axes available for plotting")

        series = pd.Series(data)
        if series.isna().any():
            raise ValueError(
                "ACFPlot got missing values; statsmodels.tsa.stattools.acf does not "
                "raise on NaN, it silently returns an all-NaN result. Drop NaNs first "
                "(a decomposition's residual has them at the edges)."
            )

        values, confint = acf(series, nlags=nlags, alpha=alpha, fft=True)  # type: ignore
        lags = np.arange(len(values))
        band = (confint[:, 1] - confint[:, 0]) / 2

        self.ax.fill_between(lags, -band, band, color=band_color, alpha=0.2)
        self.ax.axhline(0, color="black", linewidth=0.8)
        self.ax.bar(lags, values, color=color, width=0.3, **kwargs)

        return self.fig, self.ax


class ScatterPlot(BasePlot):
    """Two variables against each other. The first plot of any relation question.

    Summary statistics are not a substitute: Anscombe's quartet has four datasets
    with the same means, variances, correlation and regression line, and four
    different pictures. Look before you fit.
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        alpha: float = 0.6,
        **kwargs,
    ):
        """
        Draw a scatter of `y` against `x`.

        Parameters:
        -----------
        data : pd.DataFrame
            Frame holding both columns.
        x, y : str
            Columns to put on each axis.
        alpha : float
            Point transparency. Below 1 by default: overplotting is the standard
            way a scatter lies, and a solid blob hides its own density. On a
            large frame, sample as well — alpha alone stops helping.
        **kwargs : Additional keyword arguments passed to `sns.scatterplot`,
            including `hue`, `size` and `style`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        sns.scatterplot(data=data, x=x, y=y, alpha=alpha, ax=self.ax, **kwargs)
        return self.fig, self.ax


class RegPlot(BasePlot):
    """A fitted line through a scatter — after the scatter has told you which kind.

    `fit_reg=False` for no line, `order=2` for an obvious curve, `lowess=True` to
    let the data choose the shape. A straight line comes back with a slope and an
    r-value for a parabola and for a step function too, so the choice of fit is a
    claim about the relationship, not a formatting option.
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        fit_reg: bool = True,
        lowess: bool = False,
        order: int = 1,
        scatter: bool = True,
        color: str = "crimson",
        **kwargs,
    ):
        """
        Draw a regression fit of `y` on `x`, with the scatter under it.

        Parameters:
        -----------
        data : pd.DataFrame
            Frame holding both columns.
        x, y : str
            Columns to put on each axis.
        fit_reg : bool
            Draw the fitted line. False leaves the scatter alone.
        lowess : bool
            Fit a locally weighted curve instead of a polynomial. Seaborn draws
            no confidence band around it.
        order : int
            Polynomial order. 1 is a straight line, 2 a parabola.
        scatter : bool
            Draw the points as well. Pass False to layer a fit over an existing
            `ScatterPlot` with `plot_on`.
        color : str
            Colour of the line (and of the points, when `scatter` is True).
        **kwargs : Additional keyword arguments passed to `sns.regplot`,
            including `ci`, `robust` and `logx`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        if lowess and order != 1:
            raise ValueError(
                f"lowess=True fits its own shape, so order={order} cannot also apply; "
                "pick one."
            )

        sns.regplot(
            data=data,
            x=x,
            y=y,
            fit_reg=fit_reg,
            lowess=lowess,
            order=order,
            scatter=scatter,
            color=color,
            ax=self.ax,
            **kwargs,
        )
        return self.fig, self.ax


class CorrelationHeatmap(BasePlot):
    """A correlation matrix, for finding out where to look. Not a ranking.

    Correlations between correlated predictors are not independent evidence, so
    the strongest cell is not "the variable that matters" — it is a pair worth
    plotting. Every finding here still has to survive a scatter.
    """

    def build(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        cmap: str = "vlag",
        annot: bool = True,
        fmt: str = ".2f",
        **kwargs,
    ):
        """
        Draw the correlation matrix of a frame's numeric columns.

        Parameters:
        -----------
        data : pd.DataFrame
            Frame to correlate. Non-numeric columns are dropped.
        columns : Optional[List[str]]
            Restrict to these columns, in this order.
        method : str
            Passed to `DataFrame.corr`: `"pearson"` measures a straight-line
            relationship, `"spearman"` any monotone one. A pair that is strong
            under spearman and weak under pearson is curved, not absent.
        cmap : str
            Colormap. Diverging by default, and the scale is pinned to
            [-1, 1] around 0: correlation runs in two directions from a
            meaningful centre, and a sequential map hides that.
        annot : bool
            Write each coefficient into its cell.
        fmt : str
            Format for the annotations.
        **kwargs : Additional keyword arguments passed to `sns.heatmap`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        frame = data[columns] if columns is not None else data
        matrix = frame.corr(method=method, numeric_only=True)  # type: ignore[arg-type]
        if matrix.empty:
            raise ValueError(
                "No numeric columns to correlate; "
                f"got dtypes { ({name: str(frame[name].dtype) for name in frame}) }"
            )

        kwargs.setdefault("vmin", -1)
        kwargs.setdefault("vmax", 1)
        kwargs.setdefault("center", 0)
        sns.heatmap(matrix, cmap=cmap, annot=annot, fmt=fmt, ax=self.ax, **kwargs)
        return self.fig, self.ax


class ProjectionPlot(BasePlot):
    """Points in two dimensions after a projection has already been fitted.

    Takes coordinates, not a model: fitting PCA, t-SNE or UMAP is sklearn's job
    and stays outside this library. Pass the array it gave you.

    The axes of a t-SNE or UMAP plot mean nothing — not distance, not direction,
    not scale — so they are hidden by default. What survives is which points sit
    together, and even that is a claim to check rather than a finding.
    """

    def build(
        self,
        coordinates: Any,
        labels: Optional[Any] = None,
        alpha: float = 0.7,
        hide_axes: bool = True,
        **kwargs,
    ):
        """
        Scatter a set of two-dimensional coordinates.

        Parameters:
        -----------
        coordinates : array-like
            Shape (n, 2): the output of a fitted projection.
        labels : Optional[array-like]
            One label per point, used as colour. Length must match.
        alpha : float
            Point transparency; projections overplot heavily in dense regions.
        hide_axes : bool
            Hide the tick marks and values. The units of an embedding are not
            interpretable, and showing them invites reading a distance off them.
        **kwargs : Additional keyword arguments passed to `sns.scatterplot`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        if self.ax is None:
            raise ValueError("No axes available for plotting")

        points = np.asarray(coordinates)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"ProjectionPlot needs coordinates of shape (n, 2), got {points.shape}"
            )
        if labels is not None and len(labels) != len(points):
            raise ValueError(f"Got {len(labels)} labels for {len(points)} points")

        sns.scatterplot(
            x=points[:, 0],
            y=points[:, 1],
            hue=labels,
            alpha=alpha,
            ax=self.ax,
            **kwargs,
        )

        if hide_axes:
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        return self.fig, self.ax


class ScreePlot(BasePlot):
    """How much of the variance the components you kept actually carry.

    The figure that has to accompany a projection. Two components that explain
    12% of the variance make a picture of 12% of the data, and a shape in that
    picture is not a shape in the dataset.
    """

    def build(
        self,
        explained_variance_ratio: Any,
        n_components: Optional[int] = None,
        cumulative: bool = True,
        color: str = "steelblue",
        line_color: str = "crimson",
        **kwargs,
    ):
        """
        Draw the variance explained per component, and the running total.

        Parameters:
        -----------
        explained_variance_ratio : array-like
            One fraction per component, as `explained_variance_ratio_` gives.
        n_components : Optional[int]
            Show only the first this many components.
        cumulative : bool
            Overlay the running total, which is what "how much did I keep"
            actually asks.
        color : str
            Bar colour.
        line_color : str
            Cumulative line colour.
        **kwargs : Additional keyword arguments passed to `ax.bar`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        if self.ax is None:
            raise ValueError("No axes available for plotting")

        ratios = np.asarray(explained_variance_ratio, dtype=float)
        if ratios.ndim != 1 or len(ratios) == 0:
            raise ValueError(
                f"ScreePlot needs a one-dimensional, non-empty array of ratios, "
                f"got shape {ratios.shape}"
            )
        if n_components is not None:
            ratios = ratios[:n_components]

        components = np.arange(1, len(ratios) + 1)
        self.ax.bar(components, ratios, color=color, **kwargs)

        if cumulative:
            self.ax.plot(
                components,
                np.cumsum(ratios),
                color=line_color,
                marker="o",
                label="cumulative",
            )
            self.ax.legend()

        self.ax.set_xticks(components)
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


class QQPlot(BasePlot):
    """Quantile-quantile plot: is this fit believable, not just which one wins.

    A histogram with a pdf overlaid (PlotFits, DistPlot) is dominated by the bulk
    of the data, which is exactly where distributions look most alike. A qq-plot
    puts sorted data against the fitted distribution's theoretical quantiles, so a
    systematic deviation in a tail — where families actually differ — shows up as
    points bending away from the reference line instead of as a few stray bars.
    """

    def build(
        self,
        data: np.ndarray,
        distribution: Any,
        color: str = "steelblue",
        line_color: str = "black",
        alpha: float = 0.6,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Plot sample quantiles against a distribution's theoretical quantiles.

        Parameters:
        -----------
        data : np.ndarray
            Sample to plot
        distribution : scipy.stats distribution
            Fitted (frozen) distribution to compare against; must have a ppf method
        color : str
            Colour of the quantile points
        line_color : str
            Colour of the y=x reference line
        alpha : float
            Transparency of the quantile points
        **kwargs : Additional keyword arguments passed to ax.scatter

        Returns:
        --------
        fig, ax : The created figure and axes
        """
        if self.ax is None:
            raise ValueError("No axes available for plotting")

        sample = np.sort(np.asarray(data))
        n = len(sample)
        if n == 0:
            raise ValueError("QQPlot got no data to plot")

        # Filliben-style plotting positions: (i - 0.5) / n avoids ppf(0) / ppf(1),
        # which are -inf/inf for most families.
        probabilities = (np.arange(1, n + 1) - 0.5) / n
        theoretical = distribution.ppf(probabilities)

        self.ax.scatter(theoretical, sample, color=color, alpha=alpha, **kwargs)

        lo = min(np.min(theoretical), np.min(sample))
        hi = max(np.max(theoretical), np.max(sample))
        self.ax.plot([lo, hi], [lo, hi], color=line_color, linewidth=1, linestyle="--")

        return self.fig, self.ax


class ECDFPlot(BasePlot):
    """Empirical CDF: bin-free, and the right tool for comparing two samples.

    A histogram's shape depends on its bin width; an ECDF has no such knob, uses
    every point, and puts two samples on directly comparable axes without either
    one's binning choice hiding or manufacturing a difference.
    """

    def build(
        self,
        data: np.ndarray,
        compare: Optional[np.ndarray] = None,
        label: str = "data",
        compare_label: str = "compare",
        color: str = "steelblue",
        compare_color: str = "crimson",
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Plot the empirical CDF of `data`, and optionally a second sample over it.

        Parameters:
        -----------
        data : np.ndarray
            Sample to plot
        compare : Optional[np.ndarray]
            A second sample to overlay, for a direct two-sample comparison
        label : str
            Legend label for `data`
        compare_label : str
            Legend label for `compare`
        color : str
            Colour of the `data` step curve
        compare_color : str
            Colour of the `compare` step curve
        **kwargs : Additional keyword arguments passed to ax.step

        Returns:
        --------
        fig, ax : The created figure and axes
        """
        if self.ax is None:
            raise ValueError("No axes available for plotting")
        ax = self.ax

        self._draw(ax, np.asarray(data), color=color, label=label, **kwargs)
        if compare is not None:
            self._draw(
                ax,
                np.asarray(compare),
                color=compare_color,
                label=compare_label,
                **kwargs,
            )

        if compare is not None or label:
            ax.legend()

        return self.fig, self.ax

    def _draw(
        self, ax: Axes, values: np.ndarray, color: str, label: Optional[str], **kwargs
    ) -> None:
        if len(values) == 0:
            raise ValueError("ECDFPlot got no data to plot")
        sorted_values = np.sort(values)
        y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax.step(sorted_values, y, where="post", color=color, label=label, **kwargs)


class NullPlot(BasePlot):
    """The shuffled statistic as a histogram, with the observed value marked.

    This is what a `NullDistribution` is for. The grey cloud is what the
    measurement does when the labels mean nothing; the coloured line is what it
    did on the real data. Either the line stands outside the cloud or it does
    not, and that judgement needs no test and no critical value.
    """

    def build(
        self,
        result: NullResult,
        bins: Optional[int] = None,
        alternative: str = "two-sided",
        color: str = "lightgrey",
        line_color: str = "crimson",
        label: Optional[str] = None,
        **kwargs,
    ):
        """
        Plot a null distribution and the observed statistic over it.

        Parameters:
        -----------
        result : NullResult
            Output of `NullDistribution.run`.
        bins : Optional[int]
            Number of histogram bins; `HistogramPlot`'s default is used if None.
        alternative : str
            Passed to `NullResult.p_value` for the annotation.
        color : str
            Colour of the null histogram. Grey by default: it is the backdrop,
            not the finding.
        line_color : str
            Colour of the observed-value line.
        label : Optional[str]
            Legend label for the observed line. Built from the observed value
            and its p-value when None.
        **kwargs : Additional keyword arguments passed to `HistogramPlot`.

        Returns:
        --------
        fig, ax : The created figure and axes.
        """
        histogram = HistogramPlot(self.settings)
        self.plot_on(histogram, data=result.null, bins=bins, color=color, **kwargs)

        if label is None:
            label = self._annotation(result, alternative)

        line = VerticalLine(self.settings)
        self.plot_on(line, x=result.observed, label=label, color=line_color)

        return self.fig, self.ax

    def _annotation(self, result: NullResult, alternative: str) -> str:
        """Write the finding on the plot: the observed value and how rare it was."""
        p_value = result.p_value(alternative)
        # The count includes the observed arrangement, so p bottoms out at
        # 1/(n+1) one-sided and twice that two-sided. At the floor, the honest
        # statement is an upper bound rather than a number.
        floor = 1 / (result.n_iter + 1)
        if alternative == "two-sided":
            floor = min(1.0, 2 * floor)
        comparison = "≤" if p_value <= floor else "="
        return f"observed = {result.observed:.3g}\np {comparison} {p_value:.3g}"


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
