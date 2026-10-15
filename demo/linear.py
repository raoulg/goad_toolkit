from goad_toolkit.config import DataConfig, FileConfig
from goad_toolkit.dataprocessor import CovidDataProcessor
from goad_toolkit.models import linear_model, mse, train_model
from goad_toolkit.analytics import DistributionFitter
from loguru import logger
from goad_toolkit.visualizer import (
    ComparePlot,
    PlotSettings,
    ResidualPlot,
    PlotFits,
    FitPlotSettings,
)
import pandas as pd
import numpy as np


def preprocess() -> pd.DataFrame:
    logger.info("Start preprocessing data...")
    data_config = DataConfig()
    file_config = FileConfig()

    covidprocessor = CovidDataProcessor(file_config, data_config)
    processed = covidprocessor.process()
    logger.success("Data processed.")
    return processed


def viz_zscores(data: pd.DataFrame) -> None:
    plotsettings = PlotSettings(
        xlabel="date",
        ylabel="normalized values",
        title="Z-Scores of Deaths and Positive Tests",
    )
    compareplot = ComparePlot(plotsettings)
    compareplot.plot(
        data=data, x="date", y1="deaths_shifted_zscore", y2="positivetests_zscore"
    )
    logger.success(f"Saved {compareplot.save('data/result/zscores.png')}")


def model(data: "pd.DataFrame") -> "pd.DataFrame":
    X = data["positivetests"].values  # noqa: N806
    y = data["deaths"].values
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    initial_params = [0.01, 1.0]
    bounds = [(0, 1.0), (0, None)]
    params = train_model(X, y, linear_model, mse, initial_params, bounds=bounds)
    logger.success(f"Fitted model with parameters: {params}")
    yhat = linear_model(X, params)
    data["Predicted deaths"] = yhat
    data["residual"] = data["deaths_shifted"].values - yhat
    return data


def viz_model(data: pd.DataFrame) -> None:
    plotsettings = PlotSettings(xlabel="date", ylabel="deaths", title="Covid-19 data")
    compareplot = ComparePlot(plotsettings)
    compareplot.plot(data=data, x="date", y1="deaths_shifted", y2="Predicted deaths")
    logger.success(f"Saved {compareplot.save('data/result/linear_results.png')}")


def viz_residual(data: pd.DataFrame) -> None:
    settings = PlotSettings(
        figsize=(12, 6), title="Residual Plot", xlabel="dates", ylabel="error"
    )
    resplot = ResidualPlot(settings)
    resplot.plot(
        data=data,
        x="date",
        y="residual",
        date="2021-01-06",
        datelabel="Vaccination Started",
        interval=1,
    )
    logger.success(f"Saved {resplot.save('data/result/residuals.png')}")


def viz_distribution_fit(data: pd.DataFrame) -> None:
    fitter = DistributionFitter()
    residuals = data["residual"].to_numpy()
    fits = fitter.fit(residuals, discrete=False)
    best = fitter.best(fits)
    logger.success(f"Best fit: {best}")
    settings = PlotSettings(
        figsize=(12, 6), title="Residuals", xlabel="error", ylabel="probability"
    )
    fitplotsettings = FitPlotSettings(bins=30, max_fits=3)
    fitplotter = PlotFits(settings)
    fitplotter.plot(data=residuals, fit_results=fits, fitplotsettings=fitplotsettings)
    logger.success(f"Saved {fitplotter.save('data/result/distribution_fit.png')}")


def main() -> None:
    data = preprocess()
    viz_zscores(data)
    data = model(data)
    viz_model(data)
    viz_residual(data)
    viz_distribution_fit(data)
    logger.success("All done!")


if __name__ == "__main__":
    main()
