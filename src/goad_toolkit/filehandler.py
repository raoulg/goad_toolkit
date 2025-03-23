from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from goad_toolkit.config import FileConfig


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
        """Load data from raw directory"""
        if raw:
            subdir = "raw"
        else:
            subdir = "processed"

        if not filename:
            filename = self.config.filename

        file_path = self.config.data_dir / subdir / filename
        if not file_path.exists() and raw:
            self.download(filename=filename)

        data = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
        logger.info(f"Loaded data from {file_path}")
        return data

    def save(self, data: pd.DataFrame, filename: Optional[Path]) -> None:
        """Save data to processed directory"""
        if not filename:
            filename = self.config.filename

        processed_path = self.config.data_dir / "processed" / filename
        data.to_csv(processed_path)
        logger.success(f"Saved processed data to {processed_path}")
