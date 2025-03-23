from pathlib import Path

from pydantic import BaseModel


class FileConfig(BaseModel):
    data_dir: Path = Path.home() / ".cache/mads_datasets/covid"
    filename: Path = Path("covid.csv")
    url: str = (
        "https://raw.githubusercontent.com/mzelst/covid-19/master/data/rivm_by_day.csv"
    )

    def __repr__(self):
        return f"FileConfig(data_dir={self.data_dir},\n filename={self.filename},\nurl={self.url})"


class DataConfig(BaseModel):
    period: int = -14
    window: int = 7
    start_date: str = "2020-10-01"
    end_date: str = "2021-06-01"

    def __repr__(self):
        return f"DataConfig(period={self.period},\n window={self.window},\n start_date={self.start_date},\n end_date={self.end_date})"
