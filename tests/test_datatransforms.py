import pandas as pd
import pytest

from goad_toolkit.datatransforms import Pipeline, RegexFeature, TimeFeatures


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


@pytest.fixture
def messages() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "message": [
                "daftykins: try rebooting?",
                "see https://example.com and https://other.example",
                "no url, no mention, no question",
                None,
            ]
        }
    )


def test_count_mode(messages):
    result = RegexFeature(
        column="message", pattern=r"https?://\S+", feature="n_url", mode="count"
    )(messages.copy())

    assert result["n_url"].tolist() == [0, 2, 0, 0]


def test_has_mode(messages):
    result = RegexFeature(
        column="message", pattern=r"\?", feature="has_question", mode="has"
    )(messages.copy())

    assert result["has_question"].tolist() == [True, False, False, False]


def test_extract_mode_leaves_nan_where_nothing_matched(messages):
    result = RegexFeature(
        column="message",
        pattern=r"^(\S+)[:,]\s",
        feature="addressed_to",
        mode="extract",
    )(messages.copy())

    assert result["addressed_to"].iloc[0] == "daftykins"
    assert result["addressed_to"].isna().tolist() == [False, True, True, True]


def test_count_mode_defaults(messages):
    # mode is the one parameter with a default; it counts
    result = RegexFeature(column="message", pattern=r"\?", feature="n_question")(
        messages.copy()
    )

    assert result["n_question"].tolist() == [1, 0, 0, 0]


def test_na_rows_are_treated_as_empty_text(messages):
    result = RegexFeature(
        column="message", pattern=r"\?", feature="has_question", mode="has"
    )(messages.copy())

    # the None row must not propagate NaN into a boolean column
    assert result["has_question"].dtype == bool


def test_unknown_mode_raises(messages):
    with pytest.raises(ValueError, match="mode must be count/has/extract"):
        RegexFeature(column="message", pattern=r"\?", feature="x", mode="extractt")(
            messages.copy()
        )


def test_missing_column_raises(messages):
    with pytest.raises(ValueError, match="does not exist"):
        RegexFeature(column="nope", pattern=r"\?", feature="x")(messages.copy())


def test_extract_on_an_empty_frame_does_not_divide_by_zero():
    empty = pd.DataFrame({"message": pd.Series([], dtype="object")})

    result = RegexFeature(
        column="message", pattern=r"^(\S+):", feature="addressed_to", mode="extract"
    )(empty)

    assert len(result) == 0
    assert "addressed_to" in result.columns


def test_feature_is_the_column_name_and_name_is_the_step(messages):
    pipeline = Pipeline()
    pipeline.add(
        RegexFeature,
        name="mentions",
        column="message",
        pattern=r"^(\S+)[:,]\s",
        feature="addressed_to",
        mode="extract",
    )

    result = pipeline.apply(messages)

    assert "addressed_to" in result.columns
    assert "mentions" not in result.columns
    assert "mentions" in repr(pipeline)
