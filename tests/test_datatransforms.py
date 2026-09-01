import pandas as pd
import pytest

from goad_toolkit.datatransforms import (
    CountValues,
    Filter,
    FlagDates,
    GroupAgg,
    Head,
    MapValues,
    Pipeline,
    RegexFeature,
    Share,
    SortValues,
    SubtractBaseline,
    TimeFeatures,
    TransformBase,
)


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


class KeepFirst(TransformBase):
    """A transform that drops rows, so a report has something to report."""

    def transform(self, data: pd.DataFrame, n: int) -> pd.DataFrame:
        return data.head(n)


def test_apply_keeps_nothing_by_default(data):
    pipeline = Pipeline()
    pipeline.add(TimeFeatures, column="timestamp", features=["hour"])

    pipeline.apply(data)

    assert pipeline.intermediate == {}


def test_apply_keeps_a_frame_per_step_when_asked(data):
    pipeline = Pipeline()
    pipeline.add(TimeFeatures, name="calendar", column="timestamp", features=["hour"])
    pipeline.add(TimeFeatures, name="naming", column="timestamp", features=["day_name"])

    result = pipeline.apply(data, keep_intermediate=True)

    assert list(pipeline.intermediate) == ["calendar", "naming"]
    # Each step is kept as it stood, not as a view of the finished frame.
    assert "hour" in pipeline.intermediate["calendar"].columns
    assert "day_name" not in pipeline.intermediate["calendar"].columns
    assert "day_name" in pipeline.intermediate["naming"].columns
    assert "day_name" in result.columns


def test_keeping_intermediates_does_not_change_the_result(data):
    kept = Pipeline()
    kept.add(TimeFeatures, column="timestamp", features=["hour"])

    plain = Pipeline()
    plain.add(TimeFeatures, column="timestamp", features=["hour"])

    pd.testing.assert_frame_equal(
        kept.apply(data, keep_intermediate=True), plain.apply(data)
    )


def test_a_second_apply_replaces_the_kept_frames(data):
    pipeline = Pipeline()
    pipeline.add(TimeFeatures, column="timestamp", features=["hour"])

    pipeline.apply(data, keep_intermediate=True)
    pipeline.apply(data)

    assert pipeline.intermediate == {}


def test_report_names_the_columns_each_step_added(data):
    pipeline = Pipeline()
    pipeline.add(TimeFeatures, name="calendar", column="timestamp", features=["hour"])
    pipeline.add(TimeFeatures, name="naming", column="timestamp", features=["day_name"])

    pipeline.apply(data, keep_intermediate=True)
    report = pipeline.report()

    assert list(report["step"]) == ["calendar", "naming"]
    assert list(report["added"]) == ["hour", "day_name"]
    assert list(report["removed"]) == ["", ""]
    assert list(report["rows"]) == [len(data), len(data)]


def test_report_shows_where_the_rows_went(data):
    pipeline = Pipeline()
    pipeline.add(TimeFeatures, name="calendar", column="timestamp", features=["hour"])
    pipeline.add(KeepFirst, name="first_two", n=2)

    pipeline.apply(data, keep_intermediate=True)
    report = pipeline.report()

    assert list(report["rows"]) == [len(data), 2]


def test_report_without_intermediates_says_what_to_do():
    pipeline = Pipeline()

    with pytest.raises(ValueError, match="keep_intermediate=True"):
        pipeline.report()

@pytest.fixture
def chatlog() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]
    )
    return pd.DataFrame(
        {
            "date": dates,
            "author": ["ann", "bob", "ann", "ann", "bob"],
            "day_name": dates.day_name(),
            "n_chars": [10, 20, 30, 40, 50],
        }
    )


def test_filter_keeps_only_matching_rows(chatlog):
    result = Filter(expr="author == 'ann'")(chatlog)

    assert len(result) == 3
    assert set(result["author"]) == {"ann"}


def test_sort_values_orders_rows(chatlog):
    result = SortValues(column="n_chars", ascending=False)(chatlog)

    assert result["n_chars"].tolist() == [50, 40, 30, 20, 10]


def test_head_keeps_the_first_rows(chatlog):
    result = Head(n=2)(chatlog)

    assert len(result) == 2


def test_count_values_counts_most_frequent_first(chatlog):
    result = CountValues(column="author")(chatlog)

    assert result.columns.tolist() == ["author", "n"]
    assert result.iloc[0].tolist() == ["ann", 3]
    assert result.iloc[1].tolist() == ["bob", 2]


def test_count_values_normalizes_to_shares(chatlog):
    result = CountValues(column="author", feature="share", normalize=True)(chatlog)

    assert result["share"].sum() == pytest.approx(1.0)
    assert result.loc[result.author == "ann", "share"].iloc[0] == pytest.approx(0.6)


def test_group_agg_size_counts_rows_per_group(chatlog):
    result = GroupAgg(by="author")(chatlog)

    assert result.columns.tolist() == ["author", "n"]
    assert result.set_index("author")["n"].to_dict() == {"ann": 3, "bob": 2}


def test_group_agg_aggregates_a_column(chatlog):
    result = GroupAgg(by="author", column="n_chars", agg="sum")(chatlog)

    assert result.set_index("author")["n_chars"].to_dict() == {"ann": 80, "bob": 70}


def test_group_agg_without_a_column_rejects_other_aggregations(chatlog):
    with pytest.raises(ValueError, match="needs a column"):
        GroupAgg(by="author", agg="sum")(chatlog)


def test_share_divides_by_the_total(chatlog):
    result = Share(column="n_chars", feature="share")(chatlog.copy())

    assert result["share"].sum() == pytest.approx(1.0)
    assert "n_chars" in result.columns


def test_share_replaces_the_column_when_no_name_is_given(chatlog):
    result = Share(column="n_chars")(chatlog.copy())

    assert result["n_chars"].sum() == pytest.approx(1.0)


def test_share_normalizes_within_groups(chatlog):
    result = Share(column="n_chars", by="author", feature="share")(chatlog.copy())

    per_group = result.groupby("author")["share"].sum()
    assert per_group["ann"] == pytest.approx(1.0)
    assert per_group["bob"] == pytest.approx(1.0)


def test_flag_dates_marks_the_listed_days(chatlog):
    result = FlagDates(column="date", dates=["2024-01-02"], feature="is_event")(
        chatlog.copy()
    )

    assert result["is_event"].tolist() == [False, False, True, True, False]


def test_map_values_maps_and_fills_the_default(chatlog):
    result = MapValues(
        column="day_name",
        mapping={"Saturday": "weekend", "Sunday": "weekend"},
        feature="period",
        default="weekday",
    )(chatlog.copy())

    assert set(result["period"]) == {"weekday"}


def test_subtract_baseline_leaves_the_residual():
    observed = pd.DataFrame({"hour": [0, 1, 2], "share": [0.5, 0.3, 0.2]})
    baseline = pd.DataFrame({"hour": [0, 1, 2], "share": [0.4, 0.4, 0.2]})

    result = SubtractBaseline(column="share", baseline=baseline, on="hour")(observed)

    assert result.columns.tolist() == ["hour", "share"]
    assert result["share"].tolist() == pytest.approx([0.1, -0.1, 0.0])


def test_subtract_baseline_can_name_the_residual():
    observed = pd.DataFrame({"hour": [0, 1], "share": [0.5, 0.5]})
    baseline = pd.DataFrame({"hour": [0, 1], "share": [0.3, 0.7]})

    result = SubtractBaseline(
        column="share", baseline=baseline, on="hour", feature="difference"
    )(observed)

    assert result.columns.tolist() == ["hour", "difference"]
    assert result["difference"].tolist() == pytest.approx([0.2, -0.2])


def test_new_transforms_chain_in_a_pipeline(chatlog):
    pipeline = (
        Pipeline()
        .add(Filter, expr="n_chars >= 30")
        .add(CountValues, column="author")
        .add(SortValues, column="n", ascending=False)
        .add(Head, n=1)
    )

    result = pipeline.apply(chatlog)

    assert result["author"].tolist() == ["ann"]
    assert result["n"].tolist() == [2]


def test_flag_dates_matches_timestamps_by_calendar_day():
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(
            ["2024-01-01 09:15", "2024-01-02 23:59", "2024-01-03 00:00"]
        ).tz_localize("UTC")}
    )

    result = FlagDates(
        column="timestamp", dates=["2024-01-02"], feature="is_event"
    )(frame)

    assert result["is_event"].tolist() == [False, True, False]
