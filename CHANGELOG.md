# Changelog

## 0.2.15

### Added

- **Per-call settings overrides.** Any keyword passed to `plot()` that names a
  `PlotSettings` field (`title`, `xlabel`, `ylabel`, ...) overrides that field for that
  call only, leaving the shared settings object untouched — the claim lives in the call,
  the styling stays in the settings. On a layer whose figure already exists, the
  axis-level fields (`title`, `xlabel`, `ylabel`, `xtick_rotation`) still apply.
- `FacetPlot`: the same plot once per level of a column — small multiples in one call,
  with panel titles defaulting to the level names.
- `PlotSettings.sharex` / `sharey`: subplot grids share one scale, so panel shapes
  compare honestly. `FacetPlot` is the main customer.
- A pipeline vocabulary for the counting-and-shares work notebooks kept doing in loose
  cells: `Filter`, `SortValues`, `Head`, `CountValues`, `GroupAgg`, `Share`, `FlagDates`,
  `MapValues`, `SubtractBaseline`. Output-column parameters are spelled `feature`, the
  `RegexFeature` convention, since `name` names the pipeline step.
- `HistogramPlot.build` takes `stat` (default `"density"`), so count histograms no longer
  need a raw seaborn call.

### Fixed

- `ACFPlot` draws each lag as a vertical line (`ax.vlines`) instead of a 0.3-wide bar.
  With hundreds of lags the bars were narrower than a pixel and aliased against the
  pixel grid — lags vanished into phantom gaps or doubled up into dark stripes. Line
  width is in points, so every lag now stays visible at any `nlags`.

### Changed

- `BarPlot.build` no longer takes `xtick_rotation`; the per-call override
  (`plot(..., xtick_rotation=45)`) and the `PlotSettings` field cover both paths.

## 0.2.14

### Added

- `PlotSettings.subplot_xlabels` / `subplot_ylabels`: per-panel axis labels, the
  `subplot_titles` idea extended to labels. On the panels they cover they win over the
  figure-wide `xlabel`/`ylabel`, so a multi-panel figure with panels showing different
  quantities needs no `ax.set_ylabel` calls.
- `HorizontalLine`: a layer marking one value on the y-axis — the reference line a residual
  plot is read against. Defaults to a quiet grey line rather than `VerticalLine`'s crimson
  marker, because a baseline is context, not a finding.
- `BarPlot.build` takes `xtick_rotation` for category labels too long to sit flat.
- `PlotSettings.xtick_rotation`: the same rotation as a figure-wide setting, applied to
  every axis — for date axes whose labels overlap when flat.

### Changed

- Multi-panel grids from `create_figure` use constrained layout, which spaces panels,
  labels and the suptitle at draw time — after every layer has landed — so figures no
  longer need a trailing `fig.tight_layout()`.

## 0.2.1

### Breaking

- **`FileConfig` gained `date_column` and `date_index`.** `FileHandler.load` reads csv and
  parquet (reader picked from the suffix), parses `date_column` as dates and, when
  `date_index` is true, moves it into the index. The defaults (`"date"`, `True`) keep the
  covid pipeline working unchanged, but a `FileConfig` subclass that overrode `load` to
  support another shape can now express it as configuration instead. A file whose suffix has
  no reader, or a `date_column` that is not in the frame, raises `ValueError`. Parquet needs
  `pyarrow`, available as the `parquet` extra.
- **`DistributionRegistry` is a normal class.** Constructing one no longer returns a shared
  process-wide instance, so a registration affects only the registry it was made on. Code
  that registered a family on one registry and relied on a separately constructed
  `DistributionFitter()` picking it up must now pass it: `DistributionFitter(registry)`.

### Added

- `goad` console script: fits every family in the registry to one column of a csv or parquet
  file and prints the fits ranked by log-likelihood (`goad data/residuals.csv residual`).
- `DistributionFitter` accepts an optional `DistributionRegistry`.
- `filehandler.reader_for`, `filehandler.write_frame` and `filehandler.parse_dates` as
  reusable pieces; `FileHandler.save` writes parquet as well as csv.
- A pytest suite under `tests/`.

### Fixed

- `BarWithDates.build` and `VerticalDate.build` return `(fig, ax)` like every other `build`.
- Plots draw on `self.ax` / `self.fig` instead of module-level `plt` calls, so a layer placed
  with `plot_on_axes` lands on the axis it was handed rather than on matplotlib's current one.
- The `goad` console script points at a module that exists.

### Changed

- `__version__` is read from the installed package metadata.
