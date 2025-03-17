from typing import List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from goad.filehandler import FileHandler


class ProcessingConfig(BaseModel):
    """Configuration for data processing steps"""
    date_column: str = "date"
    value_columns: List[str] = Field(default=["deaths", "positivetests"])
    rolling_window: int = 7
    death_shift: int = -14
    scale_columns: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class DataProcessor:
    def __init__(self, filehandler: FileHandler, config: ProcessingConfig):
        self.filehandler = filehandler
        self.config = config
        self.data = None

    def load(self) -> None:
        """Load data from filehandler"""
        raw_path = self.filehandler.config.data_dir / "raw" / self.filehandler.config.filename
        self.data = pd.read_csv(raw_path)
        logger.info(f"Loaded data from {raw_path}")

    def process(self) -> pd.DataFrame:
        """Process data through all configured steps"""
        if self.data is None:
            self.load()

        # Apply processing steps
        self._convert_dates()
        self._process_deaths()
        self._filter_dates()
        self._apply_rolling_average()
        if self.config.scale_columns:
            self._scale_columns()
            
        return self.data

    def _convert_dates(self) -> None:
        """Convert date column to datetime"""
        self.data[self.config.date_column] = pd.to_datetime(
            self.data[self.config.date_column]
        )
        self.data.set_index(self.config.date_column, inplace=True)

    def _process_deaths(self) -> None:
        """Process death data with cumulative diff and shift"""
        if "deaths" in self.config.value_columns:
            self.data["deaths"] = np.concatenate([[0], np.diff(self.data["deaths"])])
            self.data["deaths"] = self.data["deaths"].shift(self.config.death_shift)

    def _filter_dates(self) -> None:
        """Filter data between start and end dates"""
        if self.config.start_date and self.config.end_date:
            mask = (
                (self.data.index > pd.to_datetime(self.config.start_date)) & 
                (self.data.index < pd.to_datetime(self.config.end_date))
            )
            self.data = self.data[mask]

    def _apply_rolling_average(self) -> None:
        """Apply rolling average to value columns"""
        self.data[self.config.value_columns] = self.data[
            self.config.value_columns
        ].rolling(self.config.rolling_window).mean()
        self.data.dropna(inplace=True)

    def _scale_columns(self) -> None:
        """Z-scale the value columns"""
        for col in self.config.value_columns:
            self.data[f"{col}_scaled"] = (
                (self.data[col] - self.data[col].mean()) / self.data[col].std()
            )

    def save(self) -> None:
        """Save processed data"""
        if self.data is not None:
            processed_path = (
                self.filehandler.config.data_dir / "processed" / self.filehandler.config.filename
            )
            self.data.to_csv(processed_path)
            logger.success(f"Saved processed data to {processed_path}")
