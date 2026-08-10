import numpy as np
import pandas as pd
import pytest

from goad_toolkit.cli import load_column, main


@pytest.fixture
def csv_file(tmp_path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"residual": rng.normal(size=200), "label": ["a"] * 200})
    path = tmp_path / "residuals.csv"
    frame.to_csv(path, index=False)
    return path


def test_load_column_drops_missing(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1.0, None, 3.0]}).to_csv(path, index=False)

    values = load_column(path, "x")

    assert list(values) == [1.0, 3.0]


def test_load_column_unknown_column(csv_file):
    with pytest.raises(SystemExit, match="Column 'nope' not found"):
        load_column(csv_file, "nope")


def test_load_column_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="No such file"):
        load_column(tmp_path / "absent.csv", "x")


def test_main_prints_a_ranked_table(csv_file, capsys):
    main([str(csv_file), "residual"])

    out = capsys.readouterr().out
    assert "200 values" in out
    assert "distribution" in out
    assert "norm" in out
    assert "likelihood" in out
