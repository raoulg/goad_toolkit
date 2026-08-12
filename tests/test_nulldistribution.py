import numpy as np
import pandas as pd
import pytest

from goad_toolkit.analytics import NullDistribution, NullResult


def mean_gap(frame: pd.DataFrame) -> float:
    """Difference in mean value between the two groups."""
    means = frame.groupby("group")["value"].mean()
    return means["a"] - means["b"]


@pytest.fixture
def separated() -> pd.DataFrame:
    """Groups that differ by ten: a gap no shuffle can reproduce."""
    return pd.DataFrame(
        {
            "group": ["a"] * 20 + ["b"] * 20,
            "value": list(np.arange(20) + 10.0) + list(np.arange(20) * 1.0),
        }
    )


@pytest.fixture
def unrelated() -> pd.DataFrame:
    """Labels assigned without reference to the values."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "group": ["a", "b"] * 30,
            "value": rng.normal(size=60),
        }
    )


def test_run_returns_observed_and_a_null_of_n_iter(separated):
    result = NullDistribution(mean_gap, n_iter=50, seed=1).run(separated, label="group")

    assert isinstance(result, NullResult)
    assert result.observed == pytest.approx(mean_gap(separated))
    assert result.null.shape == (50,)
    assert result.n_iter == 50


def test_same_seed_gives_the_same_null(separated):
    first = NullDistribution(mean_gap, n_iter=30, seed=42).run(separated, label="group")
    second = NullDistribution(mean_gap, n_iter=30, seed=42).run(
        separated, label="group"
    )

    assert np.array_equal(first.null, second.null)


def test_different_seeds_give_different_nulls(separated):
    first = NullDistribution(mean_gap, n_iter=30, seed=1).run(separated, label="group")
    second = NullDistribution(mean_gap, n_iter=30, seed=2).run(separated, label="group")

    assert not np.array_equal(first.null, second.null)


def test_the_caller_frame_is_left_alone(separated):
    before = separated.copy()

    NullDistribution(mean_gap, n_iter=20, seed=1).run(separated, label="group")

    pd.testing.assert_frame_equal(separated, before)


def test_shuffling_keeps_the_labels_and_only_breaks_the_pairing(separated):
    seen = []

    def record(frame: pd.DataFrame) -> float:
        seen.append(frame["group"].tolist())
        return mean_gap(frame)

    NullDistribution(record, n_iter=5, seed=7).run(separated, label="group")

    original = separated["group"].tolist()
    shuffles = seen[1:]  # the first call measures the data as given
    assert all(sorted(labels) == sorted(original) for labels in shuffles)
    assert any(labels != original for labels in shuffles)


def test_a_real_gap_lands_outside_the_null(separated):
    result = NullDistribution(mean_gap, n_iter=200, seed=3).run(
        separated, label="group"
    )

    assert result.observed > result.null.max()
    assert result.p_value() == pytest.approx(2 / 201)
    assert result.quantile == 1.0


def test_an_unrelated_label_lands_inside_the_null(unrelated):
    result = NullDistribution(mean_gap, n_iter=200, seed=3).run(
        unrelated, label="group"
    )

    assert result.null.min() < result.observed < result.null.max()
    assert result.p_value() > 0.05


def test_p_value_alternatives_agree_with_the_counts(unrelated):
    result = NullDistribution(mean_gap, n_iter=100, seed=5).run(
        unrelated, label="group"
    )

    greater = (np.sum(result.null >= result.observed) + 1) / 101
    less = (np.sum(result.null <= result.observed) + 1) / 101

    assert result.p_value("greater") == pytest.approx(greater)
    assert result.p_value("less") == pytest.approx(less)
    assert result.p_value("two-sided") == pytest.approx(
        min(1.0, 2 * min(greater, less))
    )


def test_p_value_never_reaches_zero(separated):
    result = NullDistribution(mean_gap, n_iter=10, seed=1).run(separated, label="group")

    assert result.p_value("greater") == pytest.approx(1 / 11)
    assert result.p_value("less") == pytest.approx(1.0)


def test_unknown_alternative_raises(separated):
    result = NullDistribution(mean_gap, n_iter=10, seed=1).run(separated, label="group")

    with pytest.raises(ValueError, match="Unknown alternative"):
        result.p_value("sideways")


def test_missing_label_column_raises(separated):
    with pytest.raises(ValueError, match="not in the data"):
        NullDistribution(mean_gap, n_iter=10).run(separated, label="cohort")


def test_n_iter_below_one_raises():
    with pytest.raises(ValueError, match="at least 1"):
        NullDistribution(mean_gap, n_iter=0)


def test_a_non_finite_observed_statistic_raises(separated):
    with pytest.raises(ValueError, match="nothing to compare"):
        NullDistribution(lambda frame: float("nan"), n_iter=10).run(
            separated, label="group"
        )


def test_repr_carries_the_numbers_worth_reading(separated):
    result = NullDistribution(mean_gap, n_iter=20, seed=1).run(separated, label="group")

    text = repr(result)
    assert "observed=" in text
    assert "quantile=" in text
    assert "n_iter=20" in text
