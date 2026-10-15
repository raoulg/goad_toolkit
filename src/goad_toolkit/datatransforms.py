from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union

import pandas as pd
from loguru import logger
from tqdm import tqdm

T = TypeVar("T", bound="TransformBase")


class TransformBase(ABC):
    """Base class for all data transformations."""

    def __init__(self, name: Optional[str] = None, **kwargs):
        self._params = kwargs
        self.name = name or self.__class__.__name__

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Checks columns, passes params to transform"""
        self._validate_column(data)

        # Call the transform method with unpacked parameters
        return self.transform(data, **self._params)

    def _validate_column(self, data: pd.DataFrame) -> None:
        """Check if the column exists in the dataframe."""
        column = self._params.get("column")
        if column and column not in data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataframe.")

    @abstractmethod
    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """Transform the data."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get the current parameters."""
        return self._params.copy()

    def update_params(self, **kwargs) -> None:
        """Update the parameters."""
        self._params.update(kwargs)

    def __repr__(self) -> str:
        """String representation of the transform."""
        params_str = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"{self.name}({params_str})"


class ShiftValues(TransformBase):
    """Shift values in a column by a specified period."""

    def transform(
        self,
        data: pd.DataFrame,
        column: str,
        period: int,
        rename: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Shift values in a column by a specified period."""
        if rename:
            colname = f"{column}_shifted"
        else:
            colname = column
        data[colname] = data[column].shift(period)
        return data


class DiffValues(TransformBase):
    """Calculate the difference between consecutive values in a column."""

    def transform(
        self, data: pd.DataFrame, column: str, rename: bool = False
    ) -> pd.DataFrame:
        """Calculate the difference between consecutive values in a column."""
        if rename:
            colname = f"{column}_diff"
        else:
            colname = column
        data[colname] = data[column].diff()
        data.iloc[0, data.columns.get_loc(colname)] = 0
        return data


class SelectDataRange(TransformBase):
    """Select rows within a specified date range."""

    def transform(
        self, data: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Select rows within a specified date range."""
        return data.loc[start_date:end_date]


class RollingAvg(TransformBase):
    """Calculate the rolling average of a column."""

    def transform(
        self, data: pd.DataFrame, column: str, window: int, rename: bool = False
    ) -> pd.DataFrame:
        """Calculate the rolling average of a column."""
        if rename:
            colname = f"{column}_rolling_avg"
        else:
            colname = column
        data[colname] = data[column].rolling(window).mean()
        data.dropna(subset=[colname], inplace=True)
        return data


class ZScaler(TransformBase):
    """Standardize the values in a column."""

    def transform(
        self, data: pd.DataFrame, column: str, rename: bool = False
    ) -> pd.DataFrame:
        """Standardize the values in a column."""
        if rename:
            colname = f"{column}_zscore"
        else:
            colname = column
        data[colname] = (data[column] - data[column].mean()) / data[column].std()
        return data


class TimeFeatures(TransformBase):
    """Derive calendar features from a timestamp column.

    `features` picks which columns to write, from `"date"`, `"hour"`,
    `"day_name"`, `"isoweek"`, `"year_week"`. Defaults to `"day_name"`,
    `"isoweek"`, `"year_week"` — the three lesson 1 already relies on.
    A feature named the same as `column` overwrites it.
    """

    AVAILABLE = ("date", "hour", "day_name", "isoweek", "year_week")
    DEFAULT = ("day_name", "isoweek", "year_week")

    def transform(
        self,
        data: pd.DataFrame,
        column: str,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        selected = list(features) if features is not None else list(self.DEFAULT)
        unknown = set(selected) - set(self.AVAILABLE)
        if unknown:
            raise ValueError(
                f"Unknown feature(s) {sorted(unknown)}; choose from {self.AVAILABLE}"
            )

        ts = pd.to_datetime(data[column])
        if "date" in selected:
            data["date"] = ts.dt.date
        if "hour" in selected:
            data["hour"] = ts.dt.hour
        if "day_name" in selected:
            data["day_name"] = ts.dt.day_name()
        if "isoweek" in selected:
            data["isoweek"] = ts.dt.isocalendar().week
        if "year_week" in selected:
            data["year_week"] = ts.dt.strftime("%Y-%W")
        return data


class RegexFeature(TransformBase):
    """Add a feature extracted from a text column with a regular expression.

    Three modes, all writing a single new column named by `feature`:

    - `"count"` — how many times the pattern occurs per row (int);
    - `"has"` — whether it occurs at all (bool);
    - `"extract"` — the first capture group's text, or NaN where nothing matched.
      `pattern` must contain exactly one capture group in this mode.

    The new column is `feature`, not `name`: `Pipeline.add(name=...)` already claims
    `name` for the step, and `TransformBase.__init__` consumes it. Any transform that
    names an output column has to spell that parameter something else.

    `mode="extract"` reports its own coverage through `loguru`. `count` and `has` fail
    visibly when a pattern is wrong — a column of all zeros or all False. Extraction
    fails silently, filling with NaN, so it says out loud how much it matched.
    """

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
            matched = data[feature].notna().sum()
            total = len(data)
            share = f"{matched / total:.1%}" if total else "n/a"
            logger.info(
                f"{self.name}: extracted '{feature}' from {matched:,}/{total:,} rows "
                f"({share}); {total - matched:,} rows had no match"
            )
        else:
            raise ValueError(f"mode must be count/has/extract, got {mode!r}")
        return data


class Filter(TransformBase):
    """Keep the rows a query expression selects.

    The expression is pandas' `DataFrame.query` syntax: column names bare,
    `and`/`or`/`not` spelled out — e.g.
    `"day_name == 'Thursday' and not is_release"`. A data decision written
    as one readable line, which is what a pipeline step should be.
    """

    def transform(self, data: pd.DataFrame, expr: str) -> pd.DataFrame:
        return data.query(expr)


class SortValues(TransformBase):
    """Sort rows by a column. Order is a data decision, so it lives here."""

    def transform(
        self, data: pd.DataFrame, column: str, ascending: bool = True
    ) -> pd.DataFrame:
        return data.sort_values(column, ascending=ascending).reset_index(drop=True)


class Head(TransformBase):
    """Keep the first `n` rows. After a sort, that is a top-n."""

    def transform(self, data: pd.DataFrame, n: int) -> pd.DataFrame:
        return data.head(n)


class CountValues(TransformBase):
    """One row per distinct value of a column, with how often it occurs.

    Sorted most-frequent first, the way `value_counts` sorts.
    `normalize=True` turns the counts into shares of the total.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str,
        feature: str = "n",
        normalize: bool = False,
    ) -> pd.DataFrame:
        counts = data[column].value_counts(normalize=normalize)
        return counts.rename(feature).rename_axis(column).reset_index()


class GroupAgg(TransformBase):
    """Aggregate a column per group: one row per group, one value each.

    `agg="size"` counts rows and needs no `column`; the count lands in
    `feature` (default `"n"`). Any other `agg` is a pandas aggregation name
    (`"sum"`, `"mean"`, `"nunique"`, ...) applied to `column`, landing in
    `feature` (default: the column's own name). The output column is
    `feature`, not `name` — `name` already names the pipeline step.
    """

    def transform(
        self,
        data: pd.DataFrame,
        by: Union[str, List[str]],
        column: Optional[str] = None,
        agg: str = "size",
        feature: Optional[str] = None,
    ) -> pd.DataFrame:
        if agg == "size":
            result = data.groupby(by).size()
            feature = feature or "n"
        else:
            if column is None:
                raise ValueError(f"agg={agg!r} needs a column to aggregate")
            result = data.groupby(by)[column].agg(agg)
            feature = feature or column
        return result.rename(feature).reset_index()


class Share(TransformBase):
    """Divide a column by its total, so the values become shares summing to one.

    With `by`, totals are per group instead: each row becomes its share of
    its own group's total, and every group sums to one separately — the move
    that makes a small group and a large one comparable by shape.
    `feature` writes the shares to a new column; by default the column is
    replaced in place.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str,
        feature: Optional[str] = None,
        by: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        if by is None:
            total = data[column].sum()
        else:
            total = data.groupby(by)[column].transform("sum")
        data[feature or column] = data[column] / total
        return data


class FlagDates(TransformBase):
    """Mark the rows that fall on the given calendar days, as a boolean column.

    The workhorse of event studies: flag release days, holidays, birthdays,
    then filter or group on the flag. Matching is by calendar day — the
    column's values are normalized to midnight, and timezone information is
    dropped, before comparing — so a timestamp column works as-is. `dates`
    is anything `pd.to_datetime` accepts.
    """

    def transform(
        self, data: pd.DataFrame, column: str, dates: Iterable[Any], feature: str
    ) -> pd.DataFrame:
        values = pd.to_datetime(data[column])
        if values.dt.tz is not None:
            values = values.dt.tz_localize(None)
        wanted = pd.to_datetime(list(dates))
        if wanted.tz is not None:
            wanted = wanted.tz_localize(None)
        data[feature] = values.dt.normalize().isin(wanted.normalize())
        return data


class MapValues(TransformBase):
    """Write a new column by mapping an existing one through a dict.

    Values missing from the mapping become `default`, so a two-entry
    mapping plus a default covers "weekend, everything else weekday".
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str,
        mapping: Dict[Any, Any],
        feature: str,
        default: Optional[Any] = None,
    ) -> pd.DataFrame:
        data[feature] = data[column].map(mapping)
        if default is not None:
            data[feature] = data[feature].fillna(default)
        return data


class SubtractBaseline(TransformBase):
    """Subtract a baseline frame's column from the same column here, keyed on `on`.

    The closing move of model-and-residual: the pipeline's frame carries the
    measured values, `baseline` carries what the model says is normal, and
    what flows on down the pipeline is the difference — the residual.
    `feature` writes the difference to a new column; by default `column` is
    replaced in place.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str,
        baseline: pd.DataFrame,
        on: str,
        feature: Optional[str] = None,
    ) -> pd.DataFrame:
        colname = feature or column
        merged = data.merge(
            baseline[[on, column]], on=on, how="left", suffixes=("", "_baseline")
        )
        merged[colname] = merged[column] - merged[f"{column}_baseline"]
        drop = [f"{column}_baseline"] + ([column] if colname != column else [])
        return merged.drop(columns=drop)


class Pipeline:
    """Pipeline for chaining data transformations."""

    def __init__(self):
        self.transforms: Dict[str, Dict[str, Any]] = {}
        self.intermediate: Dict[str, pd.DataFrame] = {}
        self.input_columns: List[str] = []

    def add(
        self, transform_class: Type[T], name: Optional[str] = None, **kwargs
    ) -> "Pipeline":
        """Add a transformation to the pipeline without instantiating it yet."""
        # Generate name if not provided
        if name is None:
            name = transform_class.__name__
            counter = 1
            while name in self.transforms:
                name = f"{name}_{counter}"
                counter += 1

        self.transforms[name] = {"class": transform_class, "params": kwargs}
        return self  # Allow method chaining

    def apply(
        self, data: pd.DataFrame, keep_intermediate: bool = False
    ) -> pd.DataFrame:
        """Apply all transformations in the pipeline.

        Args:
            data: Frame to transform. It is never modified.
            keep_intermediate: Keep the frame as it stood after every step in
                `self.intermediate`, and the input's columns in
                `self.input_columns`, so `report()` can say which step did what.
                Off by default: it holds one full copy of the frame per step.

        Returns:
            The frame after every transform, whether or not intermediates were
            kept — turning introspection on does not change what you get back.
        """
        self.intermediate = {}
        self.input_columns = list(data.columns)

        # Make a single copy at the pipeline level
        result = data.copy() if len(self.transforms) > 0 else data
        for name, transform_config in tqdm(
            self.transforms.items(), desc="Applying transforms"
        ):
            transform_class = transform_config["class"]
            params = transform_config["params"]
            transform = transform_class(name=name, **params)
            result = transform(result)
            if keep_intermediate:
                # A copy, not a reference: the steps mutate one frame in place,
                # so a stored reference would show every step holding the final
                # columns and nothing would look like it had done anything.
                self.intermediate[name] = result.copy()
        return result

    def report(self) -> pd.DataFrame:
        """One row per step: how many rows survived it, and which columns it added.

        The answer to "which step broke it", without dismantling the pipeline
        back into loose cells. Needs `apply(..., keep_intermediate=True)` first.

        Returns:
            A frame with one row per step: `step`, `rows`, `columns`, `added`
            and `removed`, in the order the steps ran.
        """
        if not self.intermediate:
            raise ValueError(
                "No intermediate frames to report on. Run "
                "apply(data, keep_intermediate=True) first."
            )

        rows = []
        previous = list(self.input_columns)
        for name, frame in self.intermediate.items():
            current = list(frame.columns)
            rows.append(
                {
                    "step": name,
                    "rows": len(frame),
                    "columns": len(current),
                    "added": ", ".join(c for c in current if c not in previous),
                    "removed": ", ".join(c for c in previous if c not in current),
                }
            )
            previous = current

        return pd.DataFrame(rows)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Get a specific transform configuration by name."""
        if key in self.transforms:
            return self.transforms[key].copy()
        raise KeyError(f"Transform '{key}' not found in pipeline")

    def __setitem__(self, key: str, params: Dict[str, Any]) -> None:
        """Update parameters for a transform using dictionary-style assignment."""
        if key in self.transforms:
            self.transforms[key]["params"].update(params)
        else:
            raise KeyError(f"Transform '{key}' not found in pipeline")

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        if not self.transforms:
            return "Pipeline(steps=[])"

        steps = []
        for name, config in self.transforms.items():
            cls = config["class"].__name__
            params = ", ".join(f"{k}={v!r}" for k, v in config["params"].items())
            steps.append(f"  {name}: {cls}({params})")

        steps_str = ",\n".join(steps)
        return f"Pipeline(\n{steps_str}\n)"
