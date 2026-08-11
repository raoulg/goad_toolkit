import pandas as pd
import pytest

from goad_toolkit.datatransforms import Pipeline, TimeFeatures


@pytest.fixture
def data() -> pd.DataFrame:
    # 2024-01-01 is a Monday
    timestamps = pd.date_range("2024-01-01 08:00", periods=10, freq="6h")
    return pd.DataFrame({"timestamp": timestamps, "value": range(10)})


def test_default_features_match_lesson_one(data):
    result = TimeFeatures(column="timestamp")(data.copy())

    assert "day_name" in result.columns
    assert "isoweek" in result.columns
    assert "year_week" in result.columns
    assert "date" not in result.columns
    assert "hour" not in result.columns


def test_selected_features_only(data):
    result = TimeFeatures(column="timestamp", features=["date", "hour"])(data.copy())

    assert "date" in result.columns
    assert "hour" in result.columns
    assert "day_name" not in result.columns
    assert result["hour"].tolist() == [8, 14, 20, 2, 8, 14, 20, 2, 8, 14]


def test_all_features(data):
    result = TimeFeatures(
        column="timestamp",
        features=list(TimeFeatures.AVAILABLE),
    )(data.copy())

    for feature in TimeFeatures.AVAILABLE:
        assert feature in result.columns
    assert result["day_name"].iloc[0] == "Monday"
    assert result["isoweek"].iloc[0] == 1


def test_unknown_feature_raises(data):
    with pytest.raises(ValueError, match="Unknown feature"):
        TimeFeatures(column="timestamp", features=["not_a_feature"])(data.copy())


def test_feature_overwrites_same_named_source_column():
    # mirrors lesson 1's own pipeline: source column is called "date"
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="D")})

    result = TimeFeatures(column="date", features=["date"])(df)

    assert result["date"].iloc[0] == pd.Timestamp("2024-01-01").date()


def test_works_in_a_pipeline(data):
    pipeline = Pipeline()
    pipeline.add(TimeFeatures, column="timestamp", features=["day_name", "hour"])

    result = pipeline.apply(data)

    assert "day_name" in result.columns
    assert "hour" in result.columns
