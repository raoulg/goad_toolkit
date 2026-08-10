from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class FileConfig(BaseModel):
    """Where the data lives and how to read it.

    `date_column` names the column to parse as dates; set it to None for data
    without a time axis. `date_index` decides whether that column becomes the
    index. The reader is picked from the file suffix (`.csv`, `.parquet`).
    """

    data_dir: Path = Path.home() / ".cache/mads_datasets/covid"
    filename: Path = Path("covid.csv")
    url: str = (
        "https://raw.githubusercontent.com/mzelst/covid-19/master/data/rivm_by_day.csv"
    )
    date_column: Optional[str] = "date"
    date_index: bool = True

    def __repr__(self):
        return (
            f"FileConfig(data_dir={self.data_dir},\n filename={self.filename},\n"
            f"url={self.url},\n date_column={self.date_column},\n"
            f" date_index={self.date_index})"
        )


class DataConfig(BaseModel):
    period: int = -14
    window: int = 7
    start_date: str = "2020-10-01"
    end_date: str = "2021-06-01"

    def __repr__(self):
        return f"DataConfig(period={self.period},\n window={self.window},\n start_date={self.start_date},\n end_date={self.end_date})"
