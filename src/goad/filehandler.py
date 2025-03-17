import requests
from loguru import logger

from goad.config import DataConfig


class FileHandler:
    def __init__(self, config: DataConfig):
        self.config = config
        self.check_dirs()

    def download(self) -> None:
        filepath = self.config.data_dir / "raw" / self.config.filename
        if not filepath.exists():
            req = requests.get(self.config.url)
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

    def load(self) -> pd.DataFrame:
        """Load data from raw directory"""
        raw_path = self.config.data_dir / "raw" / self.config.filename
        data = pd.read_csv(raw_path)
        logger.info(f"Loaded data from {raw_path}")
        return data

    def save(self, data: pd.DataFrame) -> None:
        """Save data to processed directory"""
        processed_path = self.config.data_dir / "processed" / self.config.filename
        data.to_csv(processed_path)
        logger.success(f"Saved processed data to {processed_path}")
