from pathlib import Path

from pydantic import BaseModel


class FileConfig(BaseModel):
    data_dir: Path = Path.home() / ".cache/mads_datasets/covid"
    filename: Path = Path("covid.csv")
    url: str = (
        "https://raw.githubusercontent.com/mzelst/covid-19/master/data/rivm_by_day.csv"
    )


class DataConfig(BaseModel):
    period: int = -14
    window: int = 7
    start_date: str = "2020-10-01"
    end_date: str = "2021-06-01"
