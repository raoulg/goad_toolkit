from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from goad_toolkit.config import DataConfig, FileConfig
from goad_toolkit.datatransforms import (
    DiffValues,
    Pipeline,
    RollingAvg,
    SelectDataRange,
    ShiftValues,
    ZScaler,
)
from goad_toolkit.filehandler import FileHandler


class DataProcessor(ABC):
    def __init__(self, fileconfig: FileConfig, dataconfig: DataConfig) -> None:
        self.pipeline = Pipeline()
        self.filehandler = FileHandler(fileconfig)
        self.config_pipeline(dataconfig)

    @abstractmethod
    def config_pipeline(self, dataconfig: DataConfig) -> None:
        pass

    def process(
        self, filename: Optional[Path] = None, raw: bool = True, save: bool = False
    ) -> pd.DataFrame:
        df = self.filehandler.load(filename, raw)
        result = self.pipeline.apply(df)
        if save:
            filename = filename or self.filehandler.config.filename
            self.filehandler.save(result, filename)
        return result


class CovidDataProcessor(DataProcessor):
    def config_pipeline(self, dataconfig: DataConfig) -> None:
        self.pipeline.add(DiffValues, column="deaths")
        self.pipeline.add(
            ShiftValues, column="deaths", period=dataconfig.period, rename=True
        )
        self.pipeline.add(
            SelectDataRange,
            start_date=dataconfig.start_date,
            end_date=dataconfig.end_date,
        )
        self.pipeline.add(RollingAvg, column="deaths_shifted", window=dataconfig.window)
        self.pipeline.add(RollingAvg, column="positivetests", window=dataconfig.window)
        self.pipeline.add(ZScaler, column="deaths_shifted", rename=True)
        self.pipeline.add(ZScaler, column="positivetests", rename=True)
