from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests
from loguru import logger

from goad_toolkit.config import FileConfig

READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
}


def reader_for(path: Path) -> Callable[[Path], pd.DataFrame]:
    """Pick a pandas reader based on the file suffix."""
    suffix = path.suffix.lower()
    if suffix not in READERS:
        raise ValueError(
            f"Cannot read '{path.name}': no reader for '{suffix}'. "
            f"Supported: {sorted(READERS)}"
        )
    return READERS[suffix]


def write_frame(data: pd.DataFrame, path: Path) -> None:
    """Write a dataframe, picking the writer from the file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        data.to_csv(path)
    elif suffix == ".parquet":
        data.to_parquet(path)
    else:
        raise ValueError(
            f"Cannot write '{path.name}': no writer for '{suffix}'. "
            f"Supported: ['.csv', '.parquet']"
        )


def parse_dates(
    data: pd.DataFrame, date_column: Optional[str], as_index: bool
) -> pd.DataFrame:
    """Parse `date_column` as dates, optionally moving it into the index.

    Accepts frames where the dates already are the index, so a saved frame
    round-trips through `write_frame` and back.
    """
    if date_column is None:
        return data

    if date_column in data.columns:
        data = data.assign(**{date_column: pd.to_datetime(data[date_column])})
        if as_index:
            return data.set_index(date_column)
        return data

    if data.index.name == date_column:
        data = data.set_axis(pd.to_datetime(data.index), axis=0)
        if as_index:
            return data
        return data.reset_index()

    raise ValueError(
        f"Date column '{date_column}' not found. Available columns: "
        f"{list(data.columns)}. Set FileConfig.date_column to None for data "
        "without a time axis."
    )


class FileHandler:
    def __init__(self, config: FileConfig):
        self.config = config
        self.check_dirs()

    def download(self, filename: Optional[Path] = None) -> None:
        if not filename:
            filename = self.config.filename
        filepath = self.config.data_dir / "raw" / filename
        if not filepath.exists():
            req = requests.get(self.config.url, timeout=10)
            with filepath.open("wb") as f:
                f.write(req.content)
            logger.success(f"Downloaded data to {filepath}")
        else:
            logger.info(f"{filepath} already exists\nRemove file to download again")

    def check_dirs(self) -> None:
        dirs = [
            self.config.data_dir,
            self.config.data_dir / "raw",
            self.config.data_dir / "processed",
        ]

        for dir in dirs:
            if not dir.exists():
                logger.info(f"Creating directory {dir}")
                dir.mkdir(parents=True, exist_ok=True)

    def load(self, filename: Optional[Path] = None, raw: bool = True) -> pd.DataFrame:
        """Load data from the raw or processed directory.

        The reader follows the file suffix; date handling follows
        `FileConfig.date_column` and `FileConfig.date_index`.
        """
        if raw:
            subdir = "raw"
        else:
            subdir = "processed"

        if not filename:
            filename = self.config.filename

        file_path = self.config.data_dir / subdir / filename
        if not file_path.exists() and raw:
            self.download(filename=filename)

        data = reader_for(file_path)(file_path)
        data = parse_dates(data, self.config.date_column, self.config.date_index)
        logger.info(f"Loaded data from {file_path}")
        return data

    def save(self, data: pd.DataFrame, filename: Optional[Path]) -> None:
        """Save data to processed directory"""
        if not filename:
            filename = self.config.filename

        processed_path = self.config.data_dir / "processed" / filename
        write_frame(data, processed_path)
        logger.success(f"Saved processed data to {processed_path}")
