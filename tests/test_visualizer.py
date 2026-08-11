import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from goad_toolkit.visualizer import (  # noqa: E402
    ACFPlot,
    BarWithDates,
    ComparePlot,
    ComparePlotDate,
    DecomposePlot,
    LinePlot,
    PlotSettings,
    ResidualPlot,
    VerticalDate,
)


@pytest.fixture
def data() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "deaths": range(12),
            "predicted": range(12, 24),
            "residual": [1, -1] * 6,
        }
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_lineplot_returns_fig_and_ax(data):
    fig, ax = LinePlot(PlotSettings()).plot(data=data, x="date", y="deaths")

    assert fig is not None
    assert ax is not None


def test_bar_with_dates_returns_fig_and_ax(data):
    fig, ax = BarWithDates(PlotSettings()).plot(
        data=data, x="date", y="residual", interval=1
    )

    assert fig is not None
    assert ax is not None


def test_vertical_date_returns_fig_and_ax(data):
    host = ComparePlot(PlotSettings())
    host.plot(data=data, x="date", y1="deaths", y2="predicted")

    vertical = VerticalDate(PlotSettings())
    vertical.fig, vertical.ax = host.fig, host.ax
    fig, ax = vertical.plot(date="2021-01-05", label="event")

    assert fig is host.fig
    assert ax is host.ax


def test_layers_draw_on_the_axis_they_are_handed(data):
    """A layer placed with plot_on_axes must not touch the current axes."""
    host = LinePlot(PlotSettings())
    host.create_figure(n_plots=2)
    target, other = host.fig.axes

    plt.sca(other)
    host.plot_on_axes(
        VerticalDate(PlotSettings()), target, date="2021-01-05", label="e"
    )

    assert len(target.lines) == 1
    assert len(other.lines) == 0


def test_composition_rotates_its_own_ticks(data):
    host = LinePlot(PlotSettings())
    host.create_figure(n_plots=2)
    target, other = host.fig.axes

    plt.sca(other)
    host.plot_on_axes(
        ResidualPlot(PlotSettings()),
        target,
        data=data,
        x="date",
        y="residual",
        date="2021-01-05",
        datelabel="event",
    )

    assert target.get_xticklabels()[0].get_rotation() == 45
    assert other.get_xticklabels()[0].get_rotation() == 0


def test_compare_plot_date_returns_fig_and_ax(data):
    fig, ax = ComparePlotDate(PlotSettings()).plot(
        data=data,
        x="date",
        y1="deaths",
        y2="predicted",
        date="2021-01-05",
        datelabel="event",
    )

    assert fig is not None
    assert ax is not None


@pytest.fixture
def weekly_cycle() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=56, freq="D")
    day_of_week = dates.dayofweek.to_numpy()
    seasonal = [0, 1, 2, 3, 4, -8, -8][:7]  # weekday-shaped, weekend dip
    values = [10 + i * 0.1 + seasonal[d] for i, d in enumerate(day_of_week)]
    return pd.DataFrame({"value": values}, index=dates)


def test_decompose_plot_returns_one_panel_per_component(weekly_cycle):
    fig = DecomposePlot(PlotSettings()).plot(
        data=weekly_cycle, column="value", period=7
    )

    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == 4
    titles = [ax.get_title() for ax in visible_axes]
    assert titles == ["Observed", "Trend", "Seasonal", "Residual"]


def test_acf_plot_returns_fig_and_ax(weekly_cycle):
    fig, ax = ACFPlot(PlotSettings()).plot(data=weekly_cycle["value"], nlags=14)

    assert fig is not None
    assert len(ax.patches) == 15  # lags 0..14 inclusive


def test_acf_plot_rejects_missing_values():
    series = pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0])

    with pytest.raises(ValueError, match="missing values"):
        ACFPlot(PlotSettings()).plot(data=series, nlags=2)
