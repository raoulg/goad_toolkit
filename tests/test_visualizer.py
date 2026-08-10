import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from goad_toolkit.visualizer import (  # noqa: E402
    BarWithDates,
    ComparePlot,
    ComparePlotDate,
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
