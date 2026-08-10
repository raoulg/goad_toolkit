# 3. Plot composition

## 3.1 Two different jobs, one function call

A typical plotting cell does two unrelated things at once:

```python
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df, x="date", y="deaths", ax=ax)
ax.set_title("Deaths over time")
ax.set_xlabel("date")
ax.set_ylabel("deaths per day")
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
```

One line draws the data. Six lines are makeup. They are entangled, so reusing the drawing
means copying the makeup, and changing the makeup means editing every cell that drew
anything.

GOAD splits them:

```python
from goad_toolkit.visualizer import LinePlot, PlotSettings

settings = PlotSettings(
    figsize=(12, 6),
    title="Deaths over time",
    xlabel="date",
    ylabel="deaths per day",
    grid=True,
    grid_alpha=0.3,
)
fig, ax = LinePlot(settings).plot(data=df, x="date", y="deaths")
```

`PlotSettings` is the makeup. The `BasePlot` subclass is the drawing. They travel separately,
which means a settings object can be shared across a whole figure set, and a plot class can
be reused with entirely different makeup.

## 3.2 `PlotSettings`

```python
@dataclass
class PlotSettings:
    figsize: Tuple[int, int] = (10, 6)
    title: str = "Plot"
    xlabel: str = "X"
    ylabel: str = "Y"
    legend_title: Optional[str] = None
    grid: bool = False
    grid_alpha: Optional[float] = None
    subplot_titles: List[str] = field(default_factory=list)
```

It is a dataclass, so `dataclasses.replace(settings, title="Something else")` gives you a
variant without mutating the original. Useful when a figure set shares everything but the
title.

`subplot_titles` is what makes multi-panel figures work: when a plot creates more than one
axis, each gets the corresponding entry. `PlotFits` uses this to label each fitted
distribution with its own test statistics.

## 3.3 `BasePlot`

Every plot is a subclass with one required method:

```python
class LinePlot(BasePlot):
    def build(self, data, **kwargs):
        sns.lineplot(data=data, ax=self.ax, **kwargs)
        return self.fig, self.ax
```

The contract:

- `plot(...)` is what callers use. It creates the figure if there is not one already, applies
  the settings, then calls your `build`.
- `build(...)` is what you implement. By the time it runs, `self.fig` and `self.ax` exist and
  the settings have been applied. Draw onto `self.ax`.
- Return `(self.fig, self.ax)` from `build`. Callers chain on it.

`create_figure(n_plots=...)` handles the layout: one axis for a single plot, otherwise a grid
capped at three columns with the unused axes hidden. Labels, grid and titles are applied to
every visible axis.

## 3.4 `plot_on`: layering

This is the part worth understanding properly, because it is what makes the annotation
progression in `img/example/` possible.

`plot_on(other_plot, ...)` hands another plot instance your figure and axes, then calls its
`plot` method. The result is both plots on the same axes.

```python
class ComparePlot(BasePlot):
    def build(self, data, x, y1, y2, **kwargs):
        line = LinePlot(self.settings)
        self.plot_on(line, data=data, x=x, y=y1, label=y1, **kwargs)
        self.plot_on(line, data=data, x=x, y=y2, label=y2, **kwargs)
        self.ax.tick_params(axis="x", rotation=45)
        return self.fig, self.ax
```

`ComparePlot` contains no drawing code. It is a composition of `LinePlot` called twice. That
is the whole idea: complex plots are not big functions with many branches, they are small
plots stacked.

`ResidualPlot` is the same trick with two different layers:

```python
class ResidualPlot(BasePlot):
    def build(self, data, x, y, date, datelabel, interval=1):
        self.plot_on(BarWithDates(self.settings), data=data, x=x, y=y, interval=interval)
        self.plot_on(VerticalDate(self.settings), date=date, label=datelabel)
        self.ax.tick_params(axis="x", rotation=45)
        return self.fig, self.ax
```

`VerticalDate` draws a labelled vertical line at a date. On its own it is nearly useless. As
a layer it turns any time series into an annotated one — and annotating the moment something
changed is the difference between a chart and an argument. See
[Visual critique](07-visual-critique.md) §"integrate text".

`plot_on_axes(other_plot, ax, ...)` is the same thing targeting a specific axis, which is how
multi-panel figures get built.

## 3.5 Building up a plot, one layer at a time

The three images in `img/example/` are the recommended workflow, and they are in order:

1. `base_plot.png` — the data, grey, no decoration. Does the shape carry?
2. `plot_with_trendline_and_max.png` — one analytic layer added. Does it clarify or clutter?
3. `plot_with_all_annotations.png` — text that states the finding, placed on the thing it
   refers to.

Each step is a `plot_on` call, and each step is reversible. The alternative — one giant
`sns.something(...)` call with fifteen keyword arguments — has no intermediate states to
judge, so you cannot tell which argument is earning its place.

Start at step 1 and stop when the plot answers the question. Most plots are finished before
step 3.

## 3.6 The plots that ship

| Class | Draws | Notes |
|---|---|---|
| `LinePlot` | `sns.lineplot` | the primitive most compositions build on |
| `ComparePlot` | two lines with labels | `LinePlot` twice via `plot_on` |
| `VerticalDate` | a labelled vertical line at a date | a layer, not a plot |
| `ComparePlotDate` | `ComparePlot` + `VerticalDate` | |
| `BarWithDates` | bars with a month locator on the x axis | |
| `ResidualPlot` | `BarWithDates` + `VerticalDate` | the residual-over-time view |
| `HistogramPlot` | `sns.histplot`, `stat="density"` | density so a pdf can be overlaid |
| `DistPlot` | a scipy distribution's pdf or pmf | estimates its own x-range from `ppf` |
| `PlotFits` | histogram + fitted pdf, one panel per fit | see [Distributions](05-distributions.md) |

`HistogramPlot` defaults to `sqrt(n)` bins capped at 50. That default is a starting point,
not an answer — bin count changes what a histogram appears to say, and choosing it is part of
the analysis, not a detail. Try three.

`DistPlot` takes a *frozen* scipy distribution (parameters already applied) and plots its pdf,
falling back to `pmf` for discrete families. Left to itself it picks an x-range from the
0.1st to the 99.9th percentile, which is right for most things and wrong for anything with a
very heavy tail — pass `x_range` explicitly there.

## 3.7 Writing your own plot

Two questions decide the shape:

**Is it a plot or a layer?** A plot stands alone and sets up its own axes. A layer only makes
sense on top of something else — a vertical line, a trend line, an annotation, a highlighted
point. Layers are still `BasePlot` subclasses; they are just always used through `plot_on`.

**Can it be built from existing pieces?** If yes, compose rather than draw. `ComparePlot` and
`ResidualPlot` are both zero-drawing classes, and they are the two most reused plots in the
demo.

A minimal new plot:

```python
class ScatterPlot(BasePlot):
    def build(self, data, x, y, **kwargs):
        sns.scatterplot(data=data, x=x, y=y, ax=self.ax, **kwargs)
        return self.fig, self.ax
```

A minimal new layer:

```python
class TrendLine(BasePlot):
    def build(self, data, x, y, order: int = 1, color: str = "crimson"):
        coeffs = np.polyfit(data[x], data[y], order)
        xs = np.linspace(data[x].min(), data[x].max(), 200)
        self.ax.plot(xs, np.polyval(coeffs, xs), color=color, linewidth=2)
        return self.fig, self.ax
```

```python
scatter = ScatterPlot(settings)
scatter.plot(data=df, x="weight", y="mpg", color="lightgrey")
scatter.plot_on(TrendLine(settings), data=df, x="weight", y="mpg", order=2)
```

Grey data, one coloured layer carrying the message. That is guideline five, expressed as two
lines of code.

## 3.8 Two rules that keep compositions composable

- **`build` returns `(self.fig, self.ax)`.** Every shipped plot does, layers included, so a
  caller can keep working with the figure whichever piece it got back. Yours should too.
- **Draw on `self.ax` and `self.fig`, never on `plt`.** Module-level calls like
  `plt.xticks(...)` or `plt.axvline(...)` act on matplotlib's *current* axes, which is not
  necessarily the axis your plot was handed — in a multi-panel figure it is usually the wrong
  one. The explicit forms are `self.ax.tick_params(axis="x", rotation=45)`,
  `self.ax.axvline(...)`, `self.fig.suptitle(...)`. Keeping to them is what makes
  `plot_on_axes` reliable.

`PlotFits` overrides `plot` rather than `build`, because it needs to know how many panels to
create before the figure exists. Its `build` is a no-op. It is the exception, not a pattern to
copy.

---

Next: [The five families](04-five-families.md) — choosing which plot to make.
