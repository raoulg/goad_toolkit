from goad.config import DataConfig
from goad.filehandler import FileHandler

if __name__ == "__main__":
    config = DataConfig()
    filehandler = FileHandler(config)
    filehandler.download()
