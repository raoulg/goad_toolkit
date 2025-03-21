from goad.datatransforms import (
    Pipeline,
    DiffValues,
    ShiftValues,
    SelectDataRange,
    RollingAvg,
    ZScaler,
)
from goad.config import FileConfig, DataConfig
from abc import ABC, abstractmethod
from goad.filehandler import FileHandler
import pandas as pd
from pathlib import Path
from typing import Optional


class DataProcessor(ABC):
    def __init__(self, fileconfig: FileConfig, dataconfig: DataConfig) -> None:
        self.pipeline = Pipeline()
        self.filehandler = FileHandler(fileconfig)
        self.config_pipeline(dataconfig)

    @abstractmethod
    def config_pipeline(self, dataconfig: DataConfig) -> None:
        pass

    def process(
        self, filename: Optional[Path] = None, raw: bool = True
    ) -> pd.DataFrame:
        df = self.filehandler.load(filename, raw)
        result = self.pipeline.apply(df)
        return result


class CovidDataProcessor(DataProcessor):
    def config_pipeline(self, dataconfig: DataConfig) -> None:
        self.pipeline.add(DiffValues, column="deaths")
        self.pipeline.add(ShiftValues, column="deaths", period=dataconfig.period)
        self.pipeline.add(
            SelectDataRange,
            start_date=dataconfig.start_date,
            end_date=dataconfig.end_date,
        )
        self.pipeline.add(RollingAvg, column="deaths", window=dataconfig.window)
        self.pipeline.add(ZScaler, column="deaths")
        self.pipeline.add(ZScaler, column="positivetests")
