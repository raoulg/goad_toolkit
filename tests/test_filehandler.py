import pandas as pd
import pytest

from goad_toolkit.config import FileConfig
from goad_toolkit.filehandler import FileHandler


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "deaths": [1, 2, 3],
        }
    )


def write_raw(tmp_path, frame: pd.DataFrame, name: str) -> FileConfig:
    """Put a file in the raw dir and return a config pointing at it."""
    config = FileConfig(data_dir=tmp_path, filename=name)
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "raw" / name
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return config


def test_load_csv_dates_as_index(tmp_path, frame):
    config = write_raw(tmp_path, frame, "covid.csv")
    data = FileHandler(config).load()

    assert data.index.name == "date"
    assert isinstance(data.index, pd.DatetimeIndex)
    assert list(data.columns) == ["deaths"]


def test_load_parquet(tmp_path, frame):
    config = write_raw(tmp_path, frame, "covid.parquet")
    data = FileHandler(config).load()

    assert isinstance(data.index, pd.DatetimeIndex)
    assert list(data.columns) == ["deaths"]


def test_load_custom_date_column(tmp_path, frame):
    renamed = frame.rename(columns={"date": "measured_at"})
    config = write_raw(tmp_path, renamed, "covid.csv")
    config.date_column = "measured_at"
    data = FileHandler(config).load()

    assert data.index.name == "measured_at"
    assert isinstance(data.index, pd.DatetimeIndex)


def test_load_keeps_date_as_column(tmp_path, frame):
    config = write_raw(tmp_path, frame, "covid.csv")
    config.date_index = False
    data = FileHandler(config).load()

    assert "date" in data.columns
    assert data["date"].dtype.kind == "M"
    assert data.index.name is None


def test_load_without_date_column(tmp_path):
    frame = pd.DataFrame({"weight": [1.0, 2.0], "mpg": [30.0, 25.0]})
    config = write_raw(tmp_path, frame, "cars.csv")
    config.date_column = None
    data = FileHandler(config).load()

    assert list(data.columns) == ["weight", "mpg"]


def test_load_missing_date_column_raises(tmp_path):
    frame = pd.DataFrame({"weight": [1.0, 2.0]})
    config = write_raw(tmp_path, frame, "cars.csv")

    with pytest.raises(ValueError, match="Date column 'date' not found"):
        FileHandler(config).load()


def test_load_unsupported_suffix_raises(tmp_path, frame):
    config = write_raw(tmp_path, frame, "covid.csv")
    (tmp_path / "raw" / "covid.txt").write_text("nope")

    with pytest.raises(ValueError, match="no reader for '.txt'"):
        FileHandler(config).load(filename="covid.txt")


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_save_load_roundtrip(tmp_path, frame, suffix):
    config = write_raw(tmp_path, frame, f"covid{suffix}")
    handler = FileHandler(config)
    data = handler.load()

    handler.save(data, filename=f"processed{suffix}")
    reloaded = handler.load(filename=f"processed{suffix}", raw=False)

    pd.testing.assert_frame_equal(data, reloaded)
