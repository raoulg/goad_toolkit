import goad_toolkit


def test_the_root_exports_the_four_doors():
    assert goad_toolkit.__all__ == ["DataProcessor", "Pipeline", "BasePlot", "DistributionFitter"]
    for name in goad_toolkit.__all__:
        assert getattr(goad_toolkit, name).__name__ == name
