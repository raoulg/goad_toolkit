from goad.config import DataConfig
from goad.filehandler import FileHandler
from goad.dataprocessor import (
    DataProcessor,
    ProcessingConfig,
    DateConverter,
    DeathProcessor,
    DateFilter,
    RollingAverage,
    ColumnScaler
)

if __name__ == "__main__":
    # Initialize configurations
    data_config = DataConfig()
    processing_config = ProcessingConfig(
        start_date="2020-10-01",
        end_date="2021-06-01"
    )

    # Set up pipeline
    filehandler = FileHandler(data_config)
    processing_steps = [
        DateConverter(processing_config),
        DeathProcessor(processing_config),
        DateFilter(processing_config),
        RollingAverage(processing_config),
        ColumnScaler(processing_config)
    ]
    processor = DataProcessor(processing_steps)

    # Run pipeline
    filehandler.download()
    data = filehandler.load()
    processed_data = processor.process(data)
    filehandler.save(processed_data)
