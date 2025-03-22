from goad.config import DataConfig, FileConfig
from goad.dataprocessor import CovidDataProcessor
from goad.models import linear_model, mse, train_model
from loguru import logger
from goad.visualizer import ComparePlot, PlotSettings
from pathlib import Path


def preprocess():
    logger.info("Start preprocessing data...")
    data_config = DataConfig()
    file_config = FileConfig()

    # file_handler = FileHandler(file_config)
    # raw = file_handler.load()
    covidprocessor = CovidDataProcessor(file_config, data_config)
    processed = covidprocessor.process()
    logger.success("Data processed.")
    return processed


def check_path(filepath: Path):
    parentdir = filepath.parent
    if not parentdir.exists():
        parentdir.mkdir(parents=True)
        logger.info(f"Created directory {parentdir}")


def viz_zscores(data):
    plotsettings = PlotSettings(
        xlabel="date",
        ylabel="normalized values",
        title="Z-Scores of Deaths and Positive Tests",
    )
    compareplot = ComparePlot(plotsettings)
    fig, ax = compareplot.plot(
        data, x="date", y1="deaths_shifted_zscore", y2="positivetests_zscore"
    )
    assert fig is not None, "Scores are not created"
    imgpath = "data/result/zscores.png"
    check_path(Path(imgpath))
    fig.savefig(imgpath)
    logger.success(f"Z-Scores saved to {imgpath}")


def model(data):
    X = data["positivetests"].values
    y = data["deaths"].values
    initial_params = [0.01, 1.0]
    bounds = [(0, 1.0), (0, None)]
    params = train_model(X, y, linear_model, mse, initial_params, bounds=bounds)
    return params


def viz_model(data, params):
    x = data["positivetests"].values
    yhat = linear_model(x, params)
    data["Predicted deaths"] = yhat
    plotsettings = PlotSettings(xlabel="date", ylabel="deaths", title="Covid-19 data")
    fig, ax = ComparePlot(plotsettings).plot(
        data, x="date", y1="deaths_shifted", y2="Predicted deaths"
    )

    assert fig is not None, "Model is not created"
    imgpath = "data/result/linear_results.png"
    check_path(Path(imgpath))
    fig.savefig(imgpath)
    logger.success(f"Linear model results saved to {imgpath}")


def main():
    data = preprocess()
    viz_zscores(data)
    params = model(data)
    viz_model(data, params)
    logger.success("All done!")


if __name__ == "__main__":
    main()
