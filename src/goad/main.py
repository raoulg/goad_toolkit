from goad.config import DataConfig
from goad.filehandler import FileHandler
from goad.dataprocessor import DataProcessor, ProcessingConfig

if __name__ == "__main__":
    # Initialize configurations
    data_config = DataConfig()
    processing_config = ProcessingConfig(
        start_date="2020-10-01",
        end_date="2021-06-01"
    )

    # Set up pipeline
    filehandler = FileHandler(data_config)
    processor = DataProcessor(filehandler, processing_config)

    # Run pipeline
    filehandler.download()
    processor.process()
    processor.save()
