from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
from tqdm import tqdm

import pandas as pd

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
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
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

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Shift values in a column by a specified period."""
        column = kwargs.get("column")
        period = kwargs.get("period", 0)
        rename = kwargs.get("rename", False)
        if rename:
            colname = f"{column}_shifted"
        else:
            colname = column
        data[colname] = data[column].shift(period)
        return data


class DiffValues(TransformBase):
    """Calculate the difference between consecutive values in a column."""

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate the difference between consecutive values in a column."""
        column = kwargs.get("column")
        rename = kwargs.get("rename", False)
        if rename:
            colname = f"{column}_diff"
        else:
            colname = column
        data[colname] = data[column].diff()
        data.iloc[0, data.columns.get_loc(colname)] = 0
        return data


class SelectDataRange(TransformBase):
    """Select rows within a specified date range."""

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Select rows within a specified date range."""
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        return data.loc[start_date:end_date]


class RollingAvg(TransformBase):
    """Calculate the rolling average of a column."""

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate the rolling average of a column."""
        column = kwargs.get("column")
        window = kwargs.get("window")
        if window is None:
            raise ValueError("Window size must be specified for rolling average")
        rename = kwargs.get("rename", False)
        if rename:
            colname = f"{column}_rolling_avg"
        else:
            colname = column
        data[colname] = data[column].rolling(window).mean()
        data.dropna(subset=[colname], inplace=True)
        return data


class ZScaler(TransformBase):
    """Standardize the values in a column."""

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Standardize the values in a column."""
        column = kwargs.get("column")
        rename = kwargs.get("rename", False)
        if rename:
            colname = f"{column}_zscore"
        else:
            colname = column
        data[colname] = (data[column] - data[column].mean()) / data[column].std()
        return data


class Pipeline:
    """Pipeline for chaining data transformations."""

    def __init__(self):
        self.transforms: Dict[str, Dict[str, Any]] = {}

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

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations in the pipeline."""
        # Make a single copy at the pipeline level
        result = data.copy() if len(self.transforms) > 0 else data
        for name, transform_config in tqdm(
            self.transforms.items(), desc="Applying transforms"
        ):
            transform_class = transform_config["class"]
            params = transform_config["params"]
            transform = transform_class(name=name, **params)
            result = transform(result)
        return result

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


# Example usage:
if __name__ == "__main__":
    # Sample dataframe
    df = pd.DataFrame(
        {
            "deaths": [10, 15, 20, 25, 30, 35, 40],
            "cases": [100, 150, 200, 250, 300, 350, 400],
        }
    )

    # Create pipeline
    pipeline = Pipeline()
    pipeline.add(ShiftValues, name="shift_deaths", column="deaths", period=-14)
    pipeline.add(ShiftValues, column="cases", period=-7)  # Auto-named

    # Update parameters by name
    pipeline["shift_deaths"] = {"period": -7}

    # Print pipeline
    print(pipeline)

    # Apply pipeline transformations
    result = pipeline.apply(df)

    print("\nOriginal DataFrame:")
    print(df)
    print("\nTransformed DataFrame:")
    print(result)
