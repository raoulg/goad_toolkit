from typing import List, Optional, Protocol
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field


class ProcessingStep(Protocol):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and return the modified version"""
        ...


class ProcessingConfig(BaseModel):
    """Configuration for data processing steps"""
    date_column: str = "date"
    value_columns: List[str] = Field(default=["deaths", "positivetests"])
    rolling_window: int = 7
    death_shift: int = -14
    scale_columns: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class DateConverter:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert date column to datetime and set as index"""
        data[self.config.date_column] = pd.to_datetime(data[self.config.date_column])
        data.set_index(self.config.date_column, inplace=True)
        return data


class DeathProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process death data with cumulative diff and shift"""
        if "deaths" in self.config.value_columns:
            data["deaths"] = np.concatenate([[0], np.diff(data["deaths"])])
            data["deaths"] = data["deaths"].shift(self.config.death_shift)
        return data


class DateFilter:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data between start and end dates"""
        if self.config.start_date and self.config.end_date:
            mask = (
                (data.index > pd.to_datetime(self.config.start_date)) & 
                (data.index < pd.to_datetime(self.config.end_date))
            )
            data = data[mask]
        return data


class RollingAverage:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling average to value columns"""
        data[self.config.value_columns] = data[
            self.config.value_columns
        ].rolling(self.config.rolling_window).mean()
        data.dropna(inplace=True)
        return data


class ColumnScaler:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Z-scale the value columns"""
        if self.config.scale_columns:
            for col in self.config.value_columns:
                data[f"{col}_scaled"] = (
                    (data[col] - data[col].mean()) / data[col].std()
                )
        return data


class DataProcessor:
    def __init__(self, steps: List[ProcessingStep]):
        self.steps = steps

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data through all configured steps"""
        for step in self.steps:
            data = step.process(data)
        return data
