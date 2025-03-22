# GOAD 🐐: Goal Oriented Analysis of Data

> **GOAD is the GOAT** - When your data analysis is so fire it's got rizz

GOAD🐐 is a flexible Python package for analyzing, transforming, and visualizing data with an emphasis on statistical distribution fitting and modular visualization components.

## 📊 Features

- **Composable plotting system** - Build complex visualizations by combining simple components
- **Statistical distribution fitting** - Automatically fit and compare distributions to your data
- **Data transformation pipelines** - Chain and reuse data transformations
- **Extendable architecture** - Register custom distributions and create new visualizations

> Before GOAD: mid data  
> After GOAD: data understood the assignment

## 🚀 Quick Start

### Installation

```bash
# Using uv (recommended)
uv install goad

# Or with pip
pip install goad
```

### Basic Usage

In the demo/linear.py file you can see a full demo of all the capabilities of GOAD🐐: 
- create a data processing pipeline
- components are extendable, so you can easily add your own steps to a pipeline
- create visualisations by stacking components. `BasePlot` will handle boilerplate.
- the `DistributionFitter` will try to fit a few common distributions, and add statistical tests for you
- The results work together with the `visualizer.PlotFits` class to show the results

The main strenght of this module is not that these elements are there (even thought they are very useful). Its superpower is that everything is extendable: so you can use this as a start, and extend it with your own visualisations and analytics.

> POV: Your data just got GOADed🐐 and now it's giving main character energy

## 📚 Core Components

#### 🔄 Data Transforms

GOAD🐐 provides a pipeline approach to transform your data:

```python
from goad.datatransforms import Pipeline, ShiftValues, ZScaler

# Create a pipeline
pipeline = Pipeline()

# Add transformations
pipeline.add(ShiftValues, name="shift_deaths", column="deaths", period=-14)
pipeline.add(ZScaler, name="scale_tests", column="positivetests", rename=True)

# Apply all transformations
result = pipeline.apply(data)
```

Available transforms include:
- `ShiftValues` - Shift values in a column by a specified period
- `DiffValues` - Calculate the difference between consecutive values
- `SelectDataRange` - Select rows within a specified date range
- `RollingAvg` - Calculate rolling average of a column
- `ZScaler` - Standardize values in a column

### 📊 Visualization System

GOAD🐐 visualization system is built on a composable architecture that allows you to build complex plots by combining simpler components:

```python
from goad.visualizer import PlotSettings, ResidualPlot

# Create plot settings
settings = PlotSettings(
    figsize=(12, 6), 
    title="Residual Plot", 
    xlabel="dates", 
    ylabel="error"
)

# Create residual plot (which combines other plots internally)
resplot = ResidualPlot(settings)
fig, ax = resplot.plot(
    data=data,
    x="date",
    y="residual",
    date="2021-01-06",
    datelabel="Vaccination Started",
    interval=1
)
```

The `ResidualPlot` is an example of a composite plot that combines a `BarWithDates` and `VerticalDate` plot.

### 📈 Distribution Fitting

GOAD🐐 includes tools for fitting statistical distributions to your data:

```python
from goad.analytics import DistributionFitter

# Create a fitter
fitter = DistributionFitter()

# Fit distributions to data
fits = fitter.fit(data["residual"], discrete=False)

# Get best fit(s)
best = fitter.best(fits)
print(f"Best fit: {best}")

# Visualize the fits
from goad.visualizer import PlotSettings, FitPlotSettings, PlotFits

settings = PlotSettings(
    figsize=(12, 6), 
    title="Residuals", 
    xlabel="error", 
    ylabel="probability"
)
fitplotsettings = FitPlotSettings(bins=30, max_fits=3)
fitplotter = PlotFits(settings)
fig = fitplotter.plot(
    data=data["residual"], 
    fit_results=fits, 
    fitplotsettings=fitplotsettings
)
```

### 🧩 Extending with Custom Distributions

You can easily register new distributions:

```python
from goad.distributions import DistributionRegistry
from scipy import stats

# Create registry
registry = DistributionRegistry()

# Register a new distribution
registry.register_distribution(
    name="negative_binomial", 
    dist=stats.nbinom, 
    is_discrete=True, 
    num_params=2
)

# Use in DistributionFitter
from goad.analytics import DistributionFitter
fitter = DistributionFitter()
fitter.registry = registry  # Use your custom registry
```

## 📋 Demo: Linear Model Analysis

GOAD🐐 includes a comprehensive demo that shows how to use its components together:

```python
from goad.config import DataConfig, FileConfig
from goad.dataprocessor import CovidDataProcessor
from goad.models import linear_model, mse, train_model
from goad.analytics import DistributionFitter
from goad.visualizer import ComparePlot, PlotSettings, ResidualPlot, PlotFits, FitPlotSettings

# Load and process data
data_config = DataConfig()
file_config = FileConfig()
processor = CovidDataProcessor(file_config, data_config)
data = processor.process()

# Visualize z-scores
plot_settings = PlotSettings(
    xlabel="date",
    ylabel="normalized values",
    title="Z-Scores of Deaths and Positive Tests"
)
compare_plot = ComparePlot(plot_settings)
fig, _ = compare_plot.plot(
    data=data,
    x="date",
    y1="deaths_shifted_zscore",
    y2="positivetests_zscore"
)
fig.savefig("zscores.png")

def model(data):
    # Fit linear model
    X = data["positivetests"].values
    y = data["deaths"].values
    initial_params = [0.01, 1.0]
    bounds = [(0, 1.0), (0, None)]
    params = train_model(X, y, linear_model, mse, initial_params, bounds=bounds)
    yhat = linear_model(X, params)
    data["Predicted deaths"] = yhat
    data["residual"] = data["deaths_shifted"].values - yhat
    return data

data = model(data)

# Visualize model results
fig, _ = ComparePlot(plot_settings).plot(
    data=data, 
    x="date", 
    y1="deaths_shifted", 
    y2="Predicted deaths"
)
fig.savefig("linear_results.png")

# Analyze residuals with distribution fitting
fitter = DistributionFitter()
fits = fitter.fit(data["residual"], discrete=False)
best = fitter.best(fits)

# Visualize distribution fits
settings = PlotSettings(
    figsize=(12, 6), 
    title="Residuals", 
    xlabel="error", 
    ylabel="probability"
)
fit_plot_settings = FitPlotSettings(bins=30, max_fits=3)
fit_plotter = PlotFits(settings)
fig = fit_plotter.plot(
    data=data["residual"], 
    fit_results=fits, 
    fitplotsettings=fit_plot_settings
)
fig.savefig("distribution_fit.png")
```

## 🔧 Advanced Usage: Composing Plots

GOAD🐐 has a powerful plotting system that allows you to combine plot elements:

```python
from goad.visualizer import BasePlot, LinePlot, BarWithDates, VerticalDate

# Use a base plot to create a composite
class MyCompositePlot(BasePlot):
    def build(self, data, x, y1, y2, special_date):
        # Plot the first component - a line plot
        line_plot = LinePlot(self.settings)
        self.plot_on(line_plot, data=data, x=x, y=y1, label=y1)
        
        # Plot the second component - a bar chart
        bar_plot = BarWithDates(self.settings)
        self.plot_on(bar_plot, data=data, x=x, y=y2)
        
        # Add a vertical line
        vline = VerticalDate(self.settings)
        self.plot_on(vline, date=special_date, label="Important Event")
        
        return self.fig, self.ax
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
  <em>GOAD🐐 - When your data analysis is so fire it's got rizz</em>
</p>
