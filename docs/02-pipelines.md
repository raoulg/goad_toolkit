# 2. Pipelines

## 2.1 The problem with notebook cells

A typical cleaning notebook is thirty cells of `df["x"] = ...`. It works, once, in order, on
one machine. Then:

- you rerun cell 12 after cell 20 and the frame is silently wrong;
- you get new data and cannot tell which cells were cleaning and which were exploration;
- somebody asks what you did to the data and the honest answer is "read the notebook".

A pipeline fixes all three by making the sequence an object: named steps, declared up front,
applied in order to a copy.

```python
from goad_toolkit.datatransforms import Pipeline, DiffValues, RollingAvg, ZScaler

pipeline = Pipeline()
pipeline.add(DiffValues, column="deaths")
pipeline.add(RollingAvg, name="smooth_deaths", column="deaths", window=7)
pipeline.add(ZScaler, name="scale_deaths", column="deaths", rename=True)

result = pipeline.apply(data)
```

`print(pipeline)` gives you the whole recipe, which is the answer to "what did you do to the
data".

## 2.2 How `Pipeline` works

Two details matter and both are deliberate.

**Steps are added as classes, not instances.** `pipeline.add(RollingAvg, column="deaths",
window=7)` stores the class and the keyword arguments; the transform is instantiated inside
`apply()`. This means a pipeline is a *description* that can be inspected, printed, edited
and reused, rather than a list of already-configured objects holding state.

**Parameters can be changed after the fact**, by name:

```python
pipeline["smooth_deaths"] = {"window": 14}
result = pipeline.apply(data)
```

That is what the naming is for. Give every step a name the moment you have two of the same
class in one pipeline — otherwise the auto-generated `RollingAvg_1` is what you will be
editing at midnight.

**`apply` copies once, at the pipeline level**, then lets the steps mutate that copy. So your
input frame is safe, and the steps stay cheap.

## 2.3 The transforms that ship

| Transform | Does | Key parameters |
|---|---|---|
| `ShiftValues` | shift a column by n periods | `column`, `period`, `rename` |
| `DiffValues` | difference between consecutive values | `column`, `rename` |
| `SelectDataRange` | keep rows in a date range | `start_date`, `end_date` |
| `RollingAvg` | rolling mean, drops the leading NaNs | `column`, `window`, `rename` |
| `ZScaler` | standardise to mean 0, std 1 | `column`, `rename` |

`rename=True` writes to a new column (`deaths_shifted`, `deaths_zscore`, …) instead of
overwriting. Prefer it. An overwritten column is a step you cannot debug, and the whole
argument for pipelines is that the intermediate state stays visible.

`SelectDataRange` uses `.loc[start:end]`, so it assumes a sorted datetime index. If your
frame has dates in a column rather than the index, set the index first — or write a transform
that does, which is the subject of the next section.

## 2.4 Writing your own transform

Subclass `TransformBase` and implement `transform`. That is the entire contract.

```python
from goad_toolkit.datatransforms import TransformBase
import pandas as pd

class TimeFeatures(TransformBase):
    """Derive calendar features from a timestamp column."""

    def transform(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        ts = pd.to_datetime(data[column])
        data["date"] = ts.dt.date
        data["hour"] = ts.dt.hour
        data["day_name"] = ts.dt.day_name()
        data["isoweek"] = ts.dt.isocalendar().week
        return data
```

```python
pipeline.add(TimeFeatures, column="timestamp")
```

What the base class does for you:

- stores your keyword arguments and passes them to `transform` unpacked, so your signature
  is explicit and readable rather than `**kwargs` soup;
- validates that a `column` parameter, if present, actually exists in the frame — and raises
  a message naming the column rather than a `KeyError` from six frames down;
- gives the step a `name` and a `__repr__` that shows its parameters.

Two rules for your `transform`:

- **Return the frame.** Even when you mutated it in place. The pipeline chains on the return
  value.
- **Name your parameters explicitly** in the signature. `def transform(self, data, column,
  window)` documents itself; `**kwargs` does not, and the validation cannot help you.

## 2.5 A transform for feature enrichment

The single most useful custom transform for text data, and a good template:

```python
class RegexFeature(TransformBase):
    """Add a feature extracted from a text column with a regular expression."""

    def transform(
        self,
        data: pd.DataFrame,
        column: str,
        pattern: str,
        feature: str,
        mode: str = "count",
    ) -> pd.DataFrame:
        text = data[column].fillna("")
        if mode == "count":
            data[feature] = text.str.count(pattern)
        elif mode == "has":
            data[feature] = text.str.contains(pattern, regex=True)
        elif mode == "extract":
            data[feature] = text.str.extract(pattern, expand=False)
        else:
            raise ValueError(f"mode must be count/has/extract, got {mode!r}")
        return data
```

```python
pipeline.add(RegexFeature, name="url_flag",
             column="message", pattern=r"https?://\S+", feature="has_url", mode="has")
```

Note the two different `name`s: `Pipeline.add(name=...)` names the *step*, and
`TransformBase.__init__` consumes it. So the new-column parameter has to be called something
else — `feature` here. Any transform that wants to name an output column hits this.

Enrichment before architecture is a general principle, not a text one. A hand-written
feature that encodes something you know about the domain frequently beats a more elaborate
model on the same data, and it is interpretable, and it takes four minutes.

## 2.6 `DataProcessor`: pipeline plus file handling

`DataProcessor` wires a `FileHandler` and a `Pipeline` together, so a dataset's whole path
from URL to analysis-ready frame is one class:

```python
from goad_toolkit.dataprocessor import DataProcessor

class CovidDataProcessor(DataProcessor):
    def config_pipeline(self, dataconfig):
        self.pipeline.add(DiffValues, column="deaths")
        self.pipeline.add(ShiftValues, column="deaths",
                          period=dataconfig.period, rename=True)
        self.pipeline.add(SelectDataRange,
                          start_date=dataconfig.start_date,
                          end_date=dataconfig.end_date)
        self.pipeline.add(RollingAvg, column="deaths_shifted", window=dataconfig.window)
```

```python
processor = CovidDataProcessor(FileConfig(), DataConfig())
data = processor.process()            # downloads if needed, loads, applies the pipeline
data = processor.process(save=True)   # and writes the result to data/processed/
```

Subclasses implement exactly one method: `config_pipeline`. Everything else — the download,
the directory creation, the raw/processed split, the save — is inherited.

`FileConfig` also says how the file is read: the reader follows the suffix (`.csv` or
`.parquet`), `date_column` names the column to parse as dates, and `date_index` decides
whether it becomes the index. For a dataset without a time axis, set `date_column=None`.

```python
config = FileConfig(
    data_dir=Path("data"), filename=Path("sales.parquet"), date_column="ordered_at"
)
```

The pipeline's parameters come from a pydantic `DataConfig`, which means the tuning knobs of
your analysis (the shift period, the smoothing window, the date range) live in one typed
object rather than scattered as literals through the steps. When somebody asks "what happens
if you use a 14-day window", that is a one-line change in one place.

## 2.7 Practical notes

**Order matters and is not checked.** `RollingAvg` drops rows with NaNs; running it before
`SelectDataRange` versus after gives different row counts at the boundaries. The pipeline
will not warn you. Write the steps in the order you would do them by hand.

**A pipeline is not a place for exploration.** Steps that belong in a pipeline are the ones
you would want to run again on new data. A one-off `df[df.author == "Alice"]` while you are
poking around is a notebook cell, and should stay one.

**Debugging.** When a pipeline produces something surprising, add the steps back one at a
time rather than dismantling it into cells — the moment it becomes cells again you have lost
the record of what you did, which is the thing you were trying to keep.

---

Next: [Plot composition](03-plot-composition.md) — the look step.
