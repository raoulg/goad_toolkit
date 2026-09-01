import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from goad_toolkit.analytics import NullResult  # noqa: E402
from goad_toolkit.visualizer import (  # noqa: E402
    ACFPlot,
    Annotate,
    BarbellPlot,
    BarPlot,
    BarWithDates,
    ComparePlot,
    ComparePlotDate,
    CorrelationHeatmap,
    DecomposePlot,
    ECDFPlot,
    FacetPlot,
    GroupedBarPlot,
    HeatmapPlot,
    HighlightCategory,
    HorizontalLine,
    LinePlot,
    NullPlot,
    PlotSettings,
    ProjectionPlot,
    QQPlot,
    RegPlot,
    ResidualPlot,
    ScatterPlot,
    ScreePlot,
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
    # One vlines call: a single LineCollection holding lags 0..14 inclusive.
    # Lines rather than bars so hundreds of lags cannot alias into sub-pixel
    # stripes with phantom gaps.
    lag_lines = [
        collection
        for collection in ax.collections
        if len(collection.get_paths()) == 15
    ]
    assert len(lag_lines) == 1


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


def test_horizontal_line_draws_at_the_requested_y():
    fig, ax = HorizontalLine(PlotSettings()).plot(y=0.0, label="baseline")

    assert fig is not None
    assert ax.lines[0].get_ydata()[0] == pytest.approx(0.0)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["baseline"]


def test_horizontal_line_without_a_label_has_no_legend():
    fig, ax = HorizontalLine(PlotSettings()).plot(y=1.0)

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
def categories() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": ["Adelie", "Chinstrap", "Gentoo"],
            "mass": [3700.0, 3733.0, 5076.0],
        }
    )


def facecolours(ax) -> list:
    return [patch.get_facecolor() for patch in ax.patches]


def test_create_figure_wraps_at_three_columns_by_default():
    host = LinePlot(PlotSettings())
    fig, axes = host.create_figure(n_plots=4)

    visible = [ax for ax in axes if ax.get_visible()]
    assert len(visible) == 4
    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 3)


def test_max_cols_puts_four_panels_on_one_row():
    host = LinePlot(PlotSettings(max_cols=4))
    fig, axes = host.create_figure(n_plots=4)

    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (1, 4)
    assert all(ax.get_visible() for ax in axes)


def test_max_cols_of_two_gives_a_square_grid():
    host = LinePlot(PlotSettings(max_cols=2))
    fig, axes = host.create_figure(n_plots=4)

    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 2)


def test_subplot_labels_apply_per_panel():
    settings = PlotSettings(
        subplot_xlabels=["author", "messages"],
        subplot_ylabels=["count", "length"],
    )
    host = LinePlot(settings)
    fig, axes = host.create_figure(n_plots=2)

    assert [ax.get_xlabel() for ax in axes] == ["author", "messages"]
    assert [ax.get_ylabel() for ax in axes] == ["count", "length"]


def test_xtick_rotation_setting_applies_to_every_axis():
    host = LinePlot(PlotSettings(xtick_rotation=45))
    fig, axes = host.create_figure(n_plots=2)

    fig.canvas.draw()
    for ax in axes:
        assert all(t.get_rotation() == pytest.approx(45) for t in ax.get_xticklabels())


def test_subplot_labels_win_over_the_figure_wide_label():
    settings = PlotSettings(ylabel="shared", subplot_ylabels=["only the first"])
    host = LinePlot(settings)
    fig, axes = host.create_figure(n_plots=2)

    assert axes[0].get_ylabel() == "only the first"
    assert axes[1].get_ylabel() == "shared"


def test_annotate_writes_the_text_on_the_axis():
    host = LinePlot(PlotSettings())
    host.create_figure()

    host.plot_on(Annotate(PlotSettings()), text="the dip is the outage", xy=(1.0, 2.0))

    assert [t.get_text() for t in host.ax.texts] == ["the dip is the outage"]


def test_annotate_draws_an_arrow_when_the_text_sits_elsewhere():
    host = LinePlot(PlotSettings())
    host.create_figure()

    host.plot_on(
        Annotate(PlotSettings()), text="here", xy=(1.0, 2.0), xytext=(3.0, 4.0)
    )

    annotation = host.ax.texts[0]
    assert annotation.arrow_patch is not None
    assert annotation.get_position() == (3.0, 4.0)


def test_annotate_without_a_second_point_draws_no_arrow():
    host = LinePlot(PlotSettings())
    host.create_figure()

    host.plot_on(Annotate(PlotSettings()), text="here", xy=(1.0, 2.0), arrow=True)

    assert host.ax.texts[0].arrow_patch is None


def test_projection_plot_scatters_the_coordinates():
    rng = np.random.default_rng(1)
    coordinates = rng.normal(size=(30, 2))

    fig, ax = ProjectionPlot(PlotSettings()).plot(coordinates=coordinates)

    assert fig is not None
    assert len(ax.collections[0].get_offsets()) == 30


def test_projection_plot_hides_the_meaningless_axes():
    coordinates = np.random.default_rng(1).normal(size=(10, 2))

    fig, ax = ProjectionPlot(PlotSettings()).plot(coordinates=coordinates)

    assert list(ax.get_xticks()) == []
    assert list(ax.get_yticks()) == []


def test_projection_plot_colours_by_label():
    coordinates = np.random.default_rng(1).normal(size=(6, 2))
    labels = ["a", "b"] * 3

    fig, ax = ProjectionPlot(PlotSettings()).plot(
        coordinates=coordinates, labels=labels
    )

    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["a", "b"]


def test_projection_plot_rejects_more_than_two_dimensions():
    coordinates = np.random.default_rng(1).normal(size=(10, 3))

    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        ProjectionPlot(PlotSettings()).plot(coordinates=coordinates)


def test_projection_plot_rejects_a_label_count_mismatch():
    coordinates = np.random.default_rng(1).normal(size=(10, 2))

    with pytest.raises(ValueError, match="3 labels for 10 points"):
        ProjectionPlot(PlotSettings()).plot(
            coordinates=coordinates, labels=["a", "b", "c"]
        )


def test_scree_plot_draws_a_bar_per_component_and_the_running_total():
    ratios = [0.5, 0.25, 0.15, 0.1]

    fig, ax = ScreePlot(PlotSettings()).plot(explained_variance_ratio=ratios)

    assert len(ax.patches) == 4
    cumulative = ax.lines[0].get_ydata()
    assert cumulative[-1] == pytest.approx(1.0)
    assert list(cumulative) == pytest.approx([0.5, 0.75, 0.9, 1.0])


def test_scree_plot_can_show_only_the_first_components():
    ratios = [0.5, 0.25, 0.15, 0.1]

    fig, ax = ScreePlot(PlotSettings()).plot(
        explained_variance_ratio=ratios, n_components=2
    )

    assert len(ax.patches) == 2
    assert ax.lines[0].get_ydata()[-1] == pytest.approx(0.75)


def test_scree_plot_without_the_cumulative_line():
    fig, ax = ScreePlot(PlotSettings()).plot(
        explained_variance_ratio=[0.6, 0.4], cumulative=False
    )

    assert len(ax.lines) == 0


def test_scree_plot_rejects_an_empty_array():
    with pytest.raises(ValueError, match="non-empty"):
        ScreePlot(PlotSettings()).plot(explained_variance_ratio=[])


def test_bar_plot_draws_one_bar_per_category(categories):
    fig, ax = BarPlot(PlotSettings()).plot(data=categories, x="species", y="mass")

    assert fig is not None
    assert len(ax.patches) == 3
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Adelie",
        "Chinstrap",
        "Gentoo",
    ]


def test_bar_plot_rotates_its_tick_labels_on_request(categories):
    fig, ax = BarPlot(PlotSettings()).plot(
        data=categories, x="species", y="mass", xtick_rotation=45
    )

    fig.canvas.draw()
    assert all(t.get_rotation() == pytest.approx(45) for t in ax.get_xticklabels())


def test_bar_plot_takes_an_explicit_order(categories):
    order = ["Gentoo", "Adelie", "Chinstrap"]

    fig, ax = BarPlot(PlotSettings()).plot(
        data=categories, x="species", y="mass", order=order
    )

    assert [t.get_text() for t in ax.get_xticklabels()] == order


def test_bar_plot_and_highlight_compose(categories):
    bars = BarPlot(PlotSettings(highlight=["Gentoo"]))
    bars.plot(data=categories, x="species", y="mass")

    bars.plot_on(HighlightCategory(bars.settings))

    colours = facecolours(bars.ax)
    assert colours[2] == mcolors.to_rgba("crimson")
    assert colours[0] == mcolors.to_rgba("lightgrey")


@pytest.fixture
def two_categories() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": ["Adelie", "Adelie", "Gentoo", "Gentoo"],
            "island": ["Biscoe", "Dream"] * 2,
            "mass": [3700.0, 3900.0, 5000.0, 5100.0],
        }
    )


def test_grouped_bar_plot_draws_a_bar_per_pair(two_categories):
    fig, ax = GroupedBarPlot(PlotSettings()).plot(
        data=two_categories, x="species", y="mass", hue="island"
    )

    bars = [p for p in ax.patches if p.get_height() != 0]
    assert len(bars) == 4
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["Biscoe", "Dream"]


def test_heatmap_plot_pivots_a_long_frame(two_categories):
    fig, ax = HeatmapPlot(PlotSettings()).plot(
        data=two_categories,
        index="species",
        columns="island",
        values="mass",
        aggfunc="mean",
    )

    assert [t.get_text() for t in ax.get_xticklabels()] == ["Biscoe", "Dream"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Adelie", "Gentoo"]
    assert len(ax.texts) == 4


def test_heatmap_plot_draws_a_matrix_as_it_is():
    matrix = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]], index=["a", "b"], columns=["x", "y"]
    )

    fig, ax = HeatmapPlot(PlotSettings()).plot(data=matrix)

    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]


def test_heatmap_plot_rejects_half_a_pivot(two_categories):
    with pytest.raises(ValueError, match="both index and columns"):
        HeatmapPlot(PlotSettings()).plot(data=two_categories, index="species")


def test_heatmap_plot_needs_values_to_pivot(two_categories):
    with pytest.raises(ValueError, match="values column"):
        HeatmapPlot(PlotSettings()).plot(
            data=two_categories, index="species", columns="island"
        )


@pytest.fixture
def before_after() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["a", "b", "c"],
            "before": [10.0, 20.0, 30.0],
            "after": [15.0, 18.0, 45.0],
        }
    )


def test_barbell_plot_joins_each_pair(before_after):
    fig, ax = BarbellPlot(PlotSettings()).plot(
        data=before_after, category="team", start="before", end="after"
    )

    assert fig is not None
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]
    assert len(ax.collections) == 3  # the connecting lines, plus two sets of dots
    starts = ax.collections[1].get_offsets()
    ends = ax.collections[2].get_offsets()
    assert list(starts[:, 0]) == [10.0, 20.0, 30.0]
    assert list(ends[:, 0]) == [15.0, 18.0, 45.0]


def test_barbell_plot_labels_both_ends(before_after):
    fig, ax = BarbellPlot(PlotSettings()).plot(
        data=before_after,
        category="team",
        start="before",
        end="after",
        start_label="2023",
        end_label="2024",
    )

    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["2023", "2024"]


def test_barbell_plot_names_a_missing_column(before_after):
    with pytest.raises(ValueError, match="after_2024"):
        BarbellPlot(PlotSettings()).plot(
            data=before_after, category="team", start="before", end="after_2024"
        )


def test_highlight_colours_one_category_and_greys_the_rest(categories):
    host = LinePlot(PlotSettings())
    host.create_figure()
    sns.barplot(data=categories, x="species", y="mass", ax=host.ax)

    host.plot_on(HighlightCategory(PlotSettings()), categories=["Chinstrap"])

    colours = facecolours(host.ax)
    assert colours[1] == mcolors.to_rgba("crimson")
    assert colours[0] == colours[2] == mcolors.to_rgba("lightgrey")


def test_highlight_reads_its_defaults_from_the_settings(categories):
    settings = PlotSettings(
        highlight=["Gentoo"], base_color="whitesmoke", highlight_color="navy"
    )
    host = LinePlot(settings)
    host.create_figure()
    sns.barplot(data=categories, x="species", y="mass", ax=host.ax)

    host.plot_on(HighlightCategory(settings))

    colours = facecolours(host.ax)
    assert colours[2] == mcolors.to_rgba("navy")
    assert colours[0] == mcolors.to_rgba("whitesmoke")


def test_highlight_call_arguments_win_over_the_settings(categories):
    settings = PlotSettings(highlight=["Gentoo"], highlight_color="navy")
    host = LinePlot(settings)
    host.create_figure()
    sns.barplot(data=categories, x="species", y="mass", ax=host.ax)

    host.plot_on(HighlightCategory(settings), categories=["Adelie"], color="orange")

    colours = facecolours(host.ax)
    assert colours[0] == mcolors.to_rgba("orange")
    assert colours[2] == mcolors.to_rgba("lightgrey")


def test_highlight_works_over_a_chart_pandas_drew(categories):
    host = LinePlot(PlotSettings())
    host.create_figure()
    categories.plot.bar(x="species", y="mass", ax=host.ax, legend=False)

    host.plot_on(HighlightCategory(PlotSettings()), categories=["Adelie"])

    colours = facecolours(host.ax)
    assert colours[0] == mcolors.to_rgba("crimson")
    assert colours[1] == mcolors.to_rgba("lightgrey")


def test_highlight_covers_every_bar_of_a_grouped_category():
    frame = pd.DataFrame(
        {
            "species": ["Adelie", "Adelie", "Gentoo", "Gentoo"],
            "island": ["Biscoe", "Dream"] * 2,
            "mass": [3700.0, 3900.0, 5000.0, 5100.0],
        }
    )
    host = LinePlot(PlotSettings())
    host.create_figure()
    sns.barplot(data=frame, x="species", y="mass", hue="island", ax=host.ax)

    swatches_before = [
        p.get_facecolor() for p in host.ax.patches if p.get_height() == 0
    ]

    host.plot_on(HighlightCategory(PlotSettings()), categories=["Gentoo"])

    bars = [p for p in host.ax.patches if p.get_height() != 0]
    highlighted = [p for p in bars if p.get_facecolor() == mcolors.to_rgba("crimson")]
    assert len(highlighted) == 2  # both island bars of the Gentoo group
    # The legend's proxy artists are zero-size patches on the same axis; they
    # are data-free and must keep the colours seaborn gave them.
    swatches_after = [p.get_facecolor() for p in host.ax.patches if p.get_height() == 0]
    assert swatches_after == swatches_before


def test_highlight_handles_horizontal_bars(categories):
    host = LinePlot(PlotSettings())
    host.create_figure()
    sns.barplot(data=categories, y="species", x="mass", ax=host.ax)

    host.plot_on(HighlightCategory(PlotSettings()), categories=["Gentoo"], axis="y")

    colours = facecolours(host.ax)
    assert colours[2] == mcolors.to_rgba("crimson")
    assert colours[0] == mcolors.to_rgba("lightgrey")


def test_highlight_names_the_available_categories_when_asked_for_a_missing_one(
    categories,
):
    host = LinePlot(PlotSettings())
    host.create_figure()
    sns.barplot(data=categories, x="species", y="mass", ax=host.ax)

    with pytest.raises(ValueError, match="Emperor"):
        host.plot_on(HighlightCategory(PlotSettings()), categories=["Emperor"])


def test_highlight_without_a_category_raises(categories):
    host = LinePlot(PlotSettings())
    host.create_figure()
    sns.barplot(data=categories, x="species", y="mass", ax=host.ax)

    with pytest.raises(ValueError, match="Nothing to highlight"):
        host.plot_on(HighlightCategory(PlotSettings()))


def test_highlight_on_an_empty_axis_says_to_draw_first():
    with pytest.raises(ValueError, match="draw the chart first"):
        HighlightCategory(PlotSettings()).plot(categories=["Adelie"])


def test_highlight_rejects_an_unknown_axis(categories):
    host = LinePlot(PlotSettings())
    host.create_figure()
    sns.barplot(data=categories, x="species", y="mass", ax=host.ax)

    with pytest.raises(ValueError, match="axis must be"):
        host.plot_on(HighlightCategory(PlotSettings()), categories=["Adelie"], axis="z")


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

def test_plot_call_overrides_a_settings_field(data):
    settings = PlotSettings(title="Shared title")

    fig, ax = BarPlot(settings).plot(
        data=data, x="date", y="deaths", title="This call only"
    )

    assert ax.get_title() == "This call only"


def test_plot_call_override_leaves_the_shared_settings_untouched(data):
    settings = PlotSettings(title="Shared title")

    BarPlot(settings).plot(data=data, x="date", y="deaths", title="This call only")
    fig, ax = BarPlot(settings).plot(data=data, x="date", y="deaths")

    assert settings.title == "Shared title"
    assert ax.get_title() == "Shared title"


def test_plot_call_overrides_axis_labels(data):
    fig, ax = BarPlot(PlotSettings()).plot(
        data=data, x="date", y="deaths", xlabel="when", ylabel="how many"
    )

    assert ax.get_xlabel() == "when"
    assert ax.get_ylabel() == "how many"


def test_plot_call_override_reaches_a_layer_with_an_existing_figure(data):
    settings = PlotSettings(title="Host title")
    host = BarPlot(settings)
    host.plot(data=data, x="date", y="deaths")

    host.plot_on(HorizontalLine(settings), y=0, title="Layer retitled it")

    assert host.ax.get_title() == "Layer retitled it"


def test_plot_call_override_rotates_ticks(data):
    fig, ax = BarPlot(PlotSettings()).plot(
        data=data, x="date", y="deaths", xtick_rotation=45
    )

    assert all(label.get_rotation() == 45 for label in ax.get_xticklabels())


@pytest.fixture
def facetable() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "x": list(range(4)) * 3,
            "y": range(12),
        }
    )


def test_facet_plot_draws_one_panel_per_level(facetable):
    fig, axes = FacetPlot(PlotSettings(figsize=(9, 3))).plot(
        inner=LinePlot(PlotSettings()), data=facetable, by="group", x="x", y="y"
    )

    visible = [ax for ax in axes if ax.get_visible()]
    assert len(visible) == 3
    assert [ax.get_title() for ax in visible] == ["a", "b", "c"]
    assert all(len(ax.lines) == 1 for ax in visible)


def test_facet_plot_respects_order(facetable):
    fig, axes = FacetPlot(PlotSettings()).plot(
        inner=LinePlot(PlotSettings()),
        data=facetable,
        by="group",
        order=["c", "a", "b"],
        x="x",
        y="y",
    )

    visible = [ax for ax in axes if ax.get_visible()]
    assert [ax.get_title() for ax in visible] == ["c", "a", "b"]


def test_facet_plot_shares_the_y_scale_when_asked(facetable):
    fig, axes = FacetPlot(PlotSettings(sharey=True)).plot(
        inner=LinePlot(PlotSettings()), data=facetable, by="group", x="x", y="y"
    )

    visible = [ax for ax in axes if ax.get_visible()]
    limits = {ax.get_ylim() for ax in visible}
    assert len(limits) == 1


def test_facet_plot_rejects_a_level_with_no_rows(facetable):
    with pytest.raises(ValueError, match="No rows"):
        FacetPlot(PlotSettings()).plot(
            inner=LinePlot(PlotSettings()),
            data=facetable,
            by="group",
            order=["a", "nope"],
            x="x",
            y="y",
        )


def test_facet_plot_rejects_a_missing_column(facetable):
    with pytest.raises(ValueError, match="does not exist"):
        FacetPlot(PlotSettings()).plot(
            inner=LinePlot(PlotSettings()), data=facetable, by="nope", x="x", y="y"
        )
