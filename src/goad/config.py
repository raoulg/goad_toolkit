from pathlib import Path

from pydantic import BaseModel


class DataConfig(BaseModel):
    data_dir: Path = Path.home() / ".cache/mads_datasets/covid"
    filename: Path = Path("covid.csv")
    start_date: str = "2020-10-01"
    end_date: str = "2021-06-01"
    url: str = (
        "https://raw.githubusercontent.com/mzelst/covid-19/master/data/rivm_by_day.csv"
    )
