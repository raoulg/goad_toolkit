# 8. API reference

Compact signature tables for everything the package exposes. For *why*, see the chapter each
section links to.

---

## 8.1 `goad_toolkit.config`

Pydantic settings objects. Both have a readable `__repr__`.

```python
class FileConfig(BaseModel):
    data_dir: Path = Path.home() / ".cache/mads_datasets/covid"
    filename: Path = Path("covid.csv")
    url: str = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/rivm_by_day.csv"
    date_column: Optional[str] = "date"
    date_index: bool = True

class DataConfig(BaseModel):
    period: int = -14
    window: int = 7
    start_date: str = "2020-10-01"
    end_date: str = "2021-06-01"
```

`date_column` names the column parsed as dates; `None` for data without a time axis.
`date_index` decides whether that column becomes the index.

`DataConfig` holds the tuning knobs a pipeline reads. Subclass it for your own analysis
rather than passing loose numbers into `Pipeline.add`.

---

## 8.2 `goad_toolkit.filehandler`

```python
READERS: dict[str, Callable[[Path], pd.DataFrame]]      # ".csv", ".parquet"

def reader_for(path: Path) -> Callable[[Path], pd.DataFrame]
def write_frame(data: pd.DataFrame, path: Path) -> None
def parse_dates(data: pd.DataFrame, date_column: Optional[str],
                as_index: bool) -> pd.DataFrame

class FileHandler:
    def __init__(self, config: FileConfig)
    def download(self, filename: Optional[Path] = None) -> None
    def check_dirs(self) -> None
    def load(self, filename: Optional[Path] = None, raw: bool = True) -> pd.DataFrame
    def save(self, data: pd.DataFrame, filename: Optional[Path]) -> None
```

- Creates `data_dir`, `data_dir/raw` and `data_dir/processed` on construction.
- `load(raw=True)` reads from `raw/` and downloads first if the file is missing;
  `load(raw=False)` reads from `processed/`.
- The reader and writer follow the file suffix: `.csv` and `.parquet`. Anything else raises a
  `ValueError` naming the supported suffixes. Parquet needs `pyarrow`, installed with the
  `parquet` extra: `uv add "goad-toolkit[parquet]"`.
- Date handling follows `FileConfig`: `date_column` is parsed as dates and, when
  `date_index` is true, moved into the index. A frame whose dates already *are* the index is
  accepted too, so a `save` → `load` round trip keeps its shape. A missing `date_column`
  raises a `ValueError` listing the columns that are there.

---

## 8.3 `goad_toolkit.datatransforms`

See [Pipelines](02-pipelines.md).

```python
class TransformBase(ABC):
    def __init__(self, name: Optional[str] = None, **kwargs)
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame   # validates, then calls transform
    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame   # you implement
    def get_params(self) -> Dict[str, Any]
    def update_params(self, **kwargs) -> None
```

| Transform | `transform` signature (after `data`) |
|---|---|
| `ShiftValues` | `column: str, period: int, rename: bool = False` |
| `DiffValues` | `column: str, rename: bool = False` |
| `SelectDataRange` | `start_date: str, end_date: str` |
| `RollingAvg` | `column: str, window: int, rename: bool = False` |
| `ZScaler` | `column: str, rename: bool = False` |
| `TimeFeatures` | `column: str, features: Optional[List[str]] = None` |
| `RegexFeature` | `column: str, pattern: str, feature: str, mode: str = "count"` |

`rename=True` writes to `{column}_shifted` / `_diff` / `_rolling_avg` / `_zscore` instead of
overwriting.

`TimeFeatures.AVAILABLE` is `("date", "hour", "day_name", "isoweek", "year_week")`;
`TimeFeatures.DEFAULT` — used when `features` is omitted — is `("day_name", "isoweek",
"year_week")`. A feature named the same as `column` overwrites it.

`RegexFeature.mode` is one of `"count"` (int), `"has"` (bool) or `"extract"` (the first
capture group, NaN where nothing matched); anything else raises `ValueError`. `extract`
requires exactly one capture group in `pattern` and logs its match rate through `loguru`.
The output column is `feature`, not `name` — `Pipeline.add(name=...)` already claims `name`
for the step.

```python
class Pipeline:
    def add(self, transform_class: Type[T], name: Optional[str] = None, **kwargs) -> "Pipeline"
    def apply(self, data: pd.DataFrame) -> pd.DataFrame
    def __getitem__(self, key: str) -> Dict[str, Any]
    def __setitem__(self, key: str, params: Dict[str, Any]) -> None
```

`add` takes the **class**, not an instance, and returns `self` so calls chain. `apply` copies
the frame once, then instantiates and runs each step in insertion order.

---

## 8.4 `goad_toolkit.dataprocessor`

```python
class DataProcessor(ABC):
    def __init__(self, fileconfig: FileConfig, dataconfig: DataConfig)
    def config_pipeline(self, dataconfig: DataConfig) -> None       # you implement
    def process(self, filename: Optional[Path] = None,
                raw: bool = True, save: bool = False) -> pd.DataFrame

class CovidDataProcessor(DataProcessor):   # worked example
```

Subclasses implement `config_pipeline` only. `process` loads, applies, and optionally saves.

---

## 8.5 `goad_toolkit.distributions`

```python
@dataclass
class Distribution:
    name: str
    dist: Any            # the scipy.stats distribution object
    is_discrete: bool
    num_params: int

class DistributionRegistry:
    def register_distribution(self, name: str, dist, is_discrete: bool, num_params: int)
    def get_distribution(self, name: str) -> Distribution
    def get_names(self) -> List[str]
    def is_discrete(self, name: str) -> bool
```

Registered by default: `norm`, `uniform`, `lognorm`, `poisson`, `exponential`, `skewnorm`,
`gamma`, `weibull`, `bernoulli`, `binomial`, `nbinom`, `beta`. Every instance starts from that
set, so a registration is scoped to the registry you made it on — hand that registry to
`DistributionFitter(registry)` to use it. `pareto` is not in the default set — lesson 4 has
the student register it, since a default fit missing the one family the data actually needs
is the point of that exercise.

---

## 8.6 `goad_toolkit.analytics`

See [Distributions](05-distributions.md).

```python
@dataclass
class KSTestResult:
    statistic: float
    p_value: float
    error: Optional[str] = None

@dataclass
class FitResult:
    distribution: str
    dist_object: Any
    params: Tuple[float, ...]
    frozen_dist: Any                     # ready to .pdf() / .rvs() / hand to DistPlot
    success: bool = True
    message: str = "Optimization successful"
    log_likelihood: Optional[float] = None
    kstest: Optional[KSTestResult] = None
    best_likelihood: Optional[bool] = None
    best_ks: Optional[bool] = None

@dataclass
class FailedFit:
    distribution: str
    message: str
    success: bool = False

Result = Union[FitResult, FailedFit]
```

```python
class DistributionFitter:
    def __init__(self, registry: Optional[DistributionRegistry] = None)

    registry: list[str]                                     # property, the names

    def fit(self, data: np.ndarray, discrete: bool,
            method: str = "mle",
            criterion: str = "combined") -> list[Result]

    def fit_distribution(self, dist_name: str, data: np.ndarray,
                         method: str = "mle") -> Result

    @staticmethod
    def best(results: list[Result], criterion: str = "combined") -> list[Result]

def fit_table(results: list[Result]) -> pd.DataFrame
```

`criterion` is one of `"likelihood"`, `"ks"`, `"combined"`. Failures are returned as
`FailedFit` values rather than raised. `log_likelihood` is computed with `logpdf` for
continuous families and `logpmf` for discrete ones. The two criteria are marked
independently: a `criterion="combined"` fit where every KS p-value comes back exactly 0
(routine for discrete families on a large, tied sample) still gets a `best_likelihood`
winner — it just has no `best_ks` winner.

`fit_table` turns a list of results into one row per distribution — `distribution`, `params`,
`log_likelihood`, `ks_stat`, `ks_pvalue`, `best_likelihood`, `best_ks`, `message` — ranked by
`log_likelihood` descending, with `FailedFit` rows (their `message` kept, metrics `NaN`)
sorted last.

The shuffle test — see [Models and residuals](06-models-and-residuals.md) §6.7:

```python
@dataclass
class NullResult:
    observed: float
    null: np.ndarray                     # the statistic under n_iter shuffles
    n_iter: int
    seed: Optional[int] = None

    def p_value(self, alternative: str = "two-sided") -> float
    quantile: float                      # property, where observed sits in the null

class NullDistribution:
    def __init__(self, statistic: Callable[[pd.DataFrame], float],
                 n_iter: int = 1000, seed: Optional[int] = None)

    def run(self, data: pd.DataFrame, label: str) -> NullResult
```

`statistic` takes a frame and returns one number; `run` calls it once on the data as given,
then once per shuffle of the `label` column. The caller's frame is never modified.
`alternative` is one of `"two-sided"`, `"greater"`, `"less"`. Both one-sided counts include
the observed arrangement, so `p_value` bottoms out at `1 / (n_iter + 1)` rather than 0. A
`label` that is not a column raises `ValueError`, as does a statistic that is not finite on
the unshuffled data; non-finite values among the shuffles are logged, not raised.

---

## 8.7 `goad_toolkit.models`

See [Models and residuals](06-models-and-residuals.md).

```python
def linear_model(x: np.ndarray, params: tuple | list[float]) -> np.ndarray
def logistic(x: np.ndarray, k, x0, limit=1.0) -> np.ndarray
def mse(y: np.ndarray, yhat: np.ndarray) -> float
def train_model(X, y, model_fn: Callable, loss_fn: Callable,
                params: list[float], bounds=None) -> list[float]
```

`model_fn` must have the signature `(x, params) -> yhat`. `train_model` wraps
`scipy.optimize.minimize` and returns the fitted parameter vector.

Note that `logistic` takes its parameters as separate arguments rather than as a `params`
tuple, so it is not directly usable as a `model_fn` — wrap it:

```python
def logistic_model(x, params):
    k, x0, limit = params
    return logistic(x, k, x0, limit)
```

---

## 8.8 `goad_toolkit.visualizer`

See [Plot composition](03-plot-composition.md).

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

@dataclass
class FitPlotSettings:
    bins: Optional[int] = None
    data_color: str = "lightgrey"
    best_likelihood_color: str = "crimson"
    best_ks_color: str = "darkblue"
    other_color: str = "gray"
    data_alpha: float = 0.6
    max_fits: Optional[int] = None
```

```python
class BasePlot(ABC):
    def __init__(self, settings: PlotSettings, n_plots: Optional[int] = None)
    def plot(self, n_plots: Optional[int] = None, *args, **kwargs)
    def create_figure(self, n_plots: Optional[int] = None)      # -> (fig, axes)
    def plot_on(self, other_plot: "BasePlot", *args, **kwargs)
    def plot_on_axes(self, other_plot: "BasePlot", ax: Axes, *args, **kwargs)
    def build(self, *args, **kwargs)                            # you implement
```

| Class | `build` signature (after `self`) |
|---|---|
| `LinePlot` | `data, **kwargs` → `sns.lineplot` |
| `ComparePlot` | `data, x, y1, y2, **kwargs` |
| `VerticalDate` | `date: str, label: str` |
| `VerticalLine` | `x: float, label=None, color="crimson", linestyle="--", linewidth=2, **kwargs` |
| `ComparePlotDate` | `data, x, y1, y2, date, datelabel, **kwargs` |
| `BarWithDates` | `data, x, y, interval: int = 1, **kwargs` |
| `ResidualPlot` | `data, x, y, date, datelabel, interval: int = 1` |
| `HistogramPlot` | `data: np.ndarray, bins=None, kde=False, color="skyblue", alpha=0.7, **kwargs` |
| `DistPlot` | `distribution, x_range=None, samples=1000, color="crimson", linewidth=2, label=None, **kwargs` |
| `ACFPlot` | `data, nlags=40, alpha=0.05, color="steelblue", band_color="gray", **kwargs` |
| `QQPlot` | `data, distribution, color="steelblue", line_color="black", alpha=0.6, **kwargs` |
| `ECDFPlot` | `data, compare=None, label="data", compare_label="compare", color="steelblue", compare_color="crimson", **kwargs` |
| `NullPlot` | `result: NullResult, bins=None, alternative="two-sided", color="lightgrey", line_color="crimson", label=None, **kwargs` |

```python
class PlotFits(BasePlot):
    def plot(self, data: np.ndarray,
             fit_results: List[Result],
             fitplotsettings: FitPlotSettings) -> Figure

class DecomposePlot(BasePlot):
    def plot(self, data: pd.DataFrame, column: str, period: int,
             model: str = "additive", **kwargs) -> Figure
```

Every `build` returns `(self.fig, self.ax)`, and every plot draws on `self.ax` / `self.fig`
rather than on matplotlib's current axes, so a layer placed with `plot_on_axes` lands on the
axis it was handed.

`PlotFits` overrides `plot` rather than `build`, because the panel count depends on the fit
results. `HistogramPlot` uses `stat="density"` so a pdf overlays correctly; `DistPlot` takes a
**frozen** distribution and falls back to `pmf` when there is no `pdf`.

`NullPlot` composes `HistogramPlot` and `VerticalLine`: grey cloud, coloured line. It writes
the observed value and its p-value into the legend, using `≤` when the p-value is at the
floor set by `n_iter`. Pass `label` to write your own.

---

## 8.9 `goad_toolkit.cli`

The `goad` console script fits every family in the registry to one column and prints the fits
ranked by log-likelihood.

```bash
goad data/residuals.parquet residual
goad data/counts.csv events --discrete --criterion ks
```

```python
def parse_args(argv: List[str] | None = None) -> argparse.Namespace
def load_column(file: Path, column: str) -> np.ndarray    # dropna, as an array
def format_results(results: List[Result]) -> str
def main(argv: List[str] | None = None) -> None
```

`--discrete` fits the discrete families instead of the continuous ones; `--criterion` is one
of `likelihood`, `ks`, `combined`; `--method` is passed to `scipy.stats.fit`.

---

## 8.10 Rough edges worth knowing

- `PlotFits` and `DecomposePlot` override `plot` rather than `build`; their `build` is a
  no-op. Both need the panel count decided before `create_figure` runs — from the fit results
  for `PlotFits`, always four for `DecomposePlot` — which `build` alone cannot do.
- Fits that fail come back as `FailedFit` values in the result list rather than raising, so
  a list of results can contain both kinds and needs an `isinstance` check.
- `Pipeline.apply` copies the frame once and then mutates that copy in place, so transforms
  see each other's columns — the order you add steps in matters.
- `ACFPlot` raises `ValueError` if the data contains NaNs, rather than passing them through.
  `statsmodels.tsa.stattools.acf` does not raise on missing values — it silently returns an
  all-NaN result — so a decomposition's residual needs `.dropna()` before it reaches `ACFPlot`.
- `binomial` and `nbinom` fit two shape parameters (`n`, `p`) from the sample mean and
  variance alone. If `n` is not known independently, it is not identifiable from the data —
  many `(n, p)` pairs with the same `n·p` fit about as well, so a fitted `n` should be treated
  as "consistent with the mean", not as a recovered count of trials. When you know `n` from
  the process (a fixed number of messages, a fixed number of views), trust that `n`, not the
  fitted one.

---

Next: [The analysis method](09-analysis-method.md).
