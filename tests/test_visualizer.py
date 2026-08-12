import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

from goad_toolkit.analytics import NullResult  # noqa: E402
from goad_toolkit.visualizer import (  # noqa: E402
    ACFPlot,
    BarWithDates,
    ComparePlot,
    ComparePlotDate,
    CorrelationHeatmap,
    DecomposePlot,
    ECDFPlot,
    LinePlot,
    NullPlot,
    PlotSettings,
    QQPlot,
    RegPlot,
    ResidualPlot,
    ScatterPlot,
    VerticalDate,
    VerticalLine,
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


def test_qq_plot_points_fall_near_the_reference_line_for_a_correct_fit():
    rng = np.random.default_rng(11)
    data = rng.normal(loc=5.0, scale=2.0, size=500)

    fig, ax = QQPlot(PlotSettings()).plot(
        data=data, distribution=stats.norm(loc=5.0, scale=2.0)
    )

    assert fig is not None
    offsets = ax.collections[0].get_offsets()
    theoretical, sample = offsets[:, 0], offsets[:, 1]
    assert np.corrcoef(theoretical, sample)[0, 1] > 0.99
    # A y=x reference line is drawn alongside the scatter.
    assert len(ax.lines) == 1


def test_qq_plot_rejects_empty_data():
    with pytest.raises(ValueError, match="no data"):
        QQPlot(PlotSettings()).plot(data=np.array([]), distribution=stats.norm())


def test_ecdf_plot_single_sample_is_a_valid_step_curve():
    data = np.array([1, 2, 2, 3, 5])

    fig, ax = ECDFPlot(PlotSettings()).plot(data=data)

    assert fig is not None
    line = ax.lines[0]
    y = line.get_ydata()
    assert y[-1] == pytest.approx(1.0)
    assert np.all(np.diff(y) >= 0)  # monotonically non-decreasing


def test_ecdf_plot_compares_two_samples():
    lower = np.array([1, 2, 3, 4, 5])
    higher = lower + 10

    fig, ax = ECDFPlot(PlotSettings()).plot(
        data=lower, compare=higher, label="a", compare_label="b"
    )

    assert len(ax.lines) == 2
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["a", "b"]


def test_ecdf_plot_rejects_empty_data():
    with pytest.raises(ValueError, match="no data"):
        ECDFPlot(PlotSettings()).plot(data=np.array([]))


def test_vertical_line_draws_at_the_requested_x():
    fig, ax = VerticalLine(PlotSettings()).plot(x=2.5, label="observed")

    assert fig is not None
    assert ax.lines[0].get_xdata()[0] == pytest.approx(2.5)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["observed"]


def test_vertical_line_without_a_label_has_no_legend():
    fig, ax = VerticalLine(PlotSettings()).plot(x=1.0)

    assert ax.get_legend() is None


@pytest.fixture
def null_result() -> NullResult:
    rng = np.random.default_rng(4)
    return NullResult(observed=3.0, null=rng.normal(size=200), n_iter=200, seed=4)


def test_null_plot_draws_the_null_and_marks_the_observed_value(null_result):
    fig, ax = NullPlot(PlotSettings()).plot(result=null_result)

    assert fig is not None
    assert len(ax.patches) > 0  # the null histogram
    assert ax.lines[-1].get_xdata()[0] == pytest.approx(3.0)


def test_null_plot_writes_the_finding_into_the_legend(null_result):
    fig, ax = NullPlot(PlotSettings()).plot(result=null_result)

    label = ax.get_legend().get_texts()[0].get_text()
    assert "observed = 3" in label
    assert "p" in label


def test_null_plot_reports_a_bounded_p_value_at_the_floor():
    null = np.zeros(100)
    result = NullResult(observed=10.0, null=null, n_iter=100, seed=None)

    fig, ax = NullPlot(PlotSettings()).plot(result=result)

    label = ax.get_legend().get_texts()[0].get_text()
    assert "≤" in label


@pytest.fixture
def related() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 60)
    return pd.DataFrame(
        {
            "x": x,
            "y": 3 * x + rng.normal(scale=2.0, size=60),
            "curved": x**2,
            "group": ["a", "b"] * 30,
        }
    )


def test_scatter_plot_draws_the_points(related):
    fig, ax = ScatterPlot(PlotSettings()).plot(data=related, x="x", y="y")

    assert fig is not None
    assert len(ax.collections[0].get_offsets()) == 60


def test_scatter_plot_passes_hue_through(related):
    fig, ax = ScatterPlot(PlotSettings()).plot(data=related, x="x", y="y", hue="group")

    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["a", "b"]


def test_reg_plot_draws_a_fitted_line(related):
    fig, ax = RegPlot(PlotSettings()).plot(data=related, x="x", y="y")

    assert fig is not None
    assert len(ax.lines) == 1
    slope_line = ax.lines[0].get_ydata()
    assert slope_line[-1] > slope_line[0]  # the relation is positive


def test_reg_plot_without_a_fit_draws_only_points(related):
    fig, ax = RegPlot(PlotSettings()).plot(data=related, x="x", y="y", fit_reg=False)

    assert len(ax.lines) == 0
    assert len(ax.collections[0].get_offsets()) == 60


def test_reg_plot_second_order_bends(related):
    fig, ax = RegPlot(PlotSettings()).plot(
        data=related, x="x", y="curved", order=2, scatter=False
    )

    y = ax.lines[0].get_ydata()
    # A parabola's slope changes along the line; a straight fit's does not.
    slopes = np.diff(y)
    assert slopes[-1] > slopes[0] * 2


def test_reg_plot_lowess_fits_without_a_polynomial_order(related):
    fig, ax = RegPlot(PlotSettings()).plot(
        data=related, x="x", y="y", lowess=True, scatter=False
    )

    assert len(ax.lines) == 1


def test_reg_plot_rejects_lowess_with_an_order(related):
    with pytest.raises(ValueError, match="pick one"):
        RegPlot(PlotSettings()).plot(data=related, x="x", y="y", lowess=True, order=2)


def test_reg_plot_layers_onto_a_scatter(related):
    scatter = ScatterPlot(PlotSettings())
    scatter.plot(data=related, x="x", y="y")

    scatter.plot_on(RegPlot(PlotSettings()), data=related, x="x", y="y", scatter=False)

    assert len(scatter.ax.lines) == 1
    # The scatter's own points, plus the fit's confidence band — but no second
    # set of points: scatter=False left the drawing of them to ScatterPlot.
    point_sets = [c for c in scatter.ax.collections if len(c.get_offsets()) == 60]
    assert len(point_sets) == 1


def test_correlation_heatmap_covers_every_numeric_pair(related):
    fig, ax = CorrelationHeatmap(PlotSettings()).plot(data=related)

    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["x", "y", "curved"]  # "group" is not numeric
    assert len(ax.texts) == 9  # one annotation per cell


def test_correlation_heatmap_pins_the_scale_to_the_full_range(related):
    fig, ax = CorrelationHeatmap(PlotSettings()).plot(data=related)

    assert ax.collections[0].get_clim() == (-1, 1)


def test_correlation_heatmap_takes_a_column_subset(related):
    fig, ax = CorrelationHeatmap(PlotSettings()).plot(data=related, columns=["y", "x"])

    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["y", "x"]


def test_correlation_heatmap_reports_a_curve_under_spearman():
    frame = pd.DataFrame({"x": np.linspace(1, 10, 50)})
    frame["exp"] = np.exp(frame["x"])

    fig, ax = CorrelationHeatmap(PlotSettings()).plot(data=frame, method="spearman")

    annotations = sorted(float(t.get_text()) for t in ax.texts)
    assert annotations == [1.0, 1.0, 1.0, 1.0]  # monotone, though far from linear


def test_correlation_heatmap_rejects_a_frame_with_nothing_numeric():
    frame = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})

    with pytest.raises(ValueError, match="No numeric columns"):
        CorrelationHeatmap(PlotSettings()).plot(data=frame)


def test_null_plot_accepts_a_custom_label(null_result):
    fig, ax = NullPlot(PlotSettings()).plot(result=null_result, label="my finding")

    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["my finding"]
