# 4. The five families

Almost every question you can ask of a dataset falls into one of five families. Each family
has a small set of plots that answer it, and — more usefully — a characteristic way of lying.

Pick the family from the **question**, not from the chart you already pictured.

| If your question is about… | Family | Reach for |
|---|---|---|
| how groups differ | categories | bar, grouped bar, barbell, heatmap |
| how something changes over time | time series | line, area, decomposition, autocorrelation |
| how a single variable is spread | distributions | histogram, ECDF, qq-plot, fitted pdf |
| whether two variables move together | relations | scatter, regression fit, correlation matrix |
| structure in many variables at once | high-dimensional | projection (PCA/t-SNE), scree |

---

## 4.1 Categories — how do groups differ?

**The question sounds like:** who sends more, which day is busiest, does group A differ from
group B.

**The plots:** a bar chart for counts or means per group; a grouped or stacked bar for two
categorical variables at once; a heatmap when both categorical variables have many levels; a
barbell when each group has a before and an after and the *change* is the point.

**How it lies:**

- **Unordered bars.** Alphabetical order is almost never meaningful order. Sort by the value
  unless the category has an inherent sequence (days of the week, age bands). An unsorted bar
  chart forces the reader to do the comparison you were supposed to do for them.
- **Proportions across wildly unequal groups.** "40% of group A versus 25% of group B" is a
  different statement when A has 2000 members and B has 8. Show the counts, or the
  uncertainty, or both.
- **Simpson's paradox.** The aggregate comparison and every within-group comparison point in
  opposite directions. This is not a rare pathology — it happens whenever the groups differ
  in composition, which is usually. The defence is to ask, every time: *is there a variable
  that differs between these groups and also affects the outcome?* If yes, compare within it.
- **Too many categories.** Past roughly a dozen bars, nobody is comparing anything. Take the
  top k and aggregate the rest into "other" — and say that you did.
- **Colour used as decoration.** A different colour per bar when the bars are already
  labelled adds nothing and costs attention. Grey, plus colour on the one bar you are talking
  about.

**In GOAD:** `BarPlot`, `GroupedBarPlot`, `HeatmapPlot` and `BarbellPlot`, with
`HighlightCategory` layered on top when one category is the point.

The category counts come from a `Pipeline` (`groupby` in a transform). Ordering and the top-k
cut belong in the pipeline too, so they are recorded, not in the plot call, where they are
invisible — which is why `BarPlot` draws the frame in the order it is handed and offers no
`sort` of its own.

The four are the same question at different sizes: one categorical variable is `BarPlot`, two
is `GroupedBarPlot` until the legend gets long, and past that it is `HeatmapPlot`.
`BarbellPlot` is the case where each group has a before and an after — two bar charts make the
reader do the subtraction, a barbell draws it.

Lesson 2 has you write a `BarPlot` of your own before any of this. That exercise is about the
`BasePlot` extension point, and it is not made pointless by the library shipping one, any more
than writing a sort is made pointless by `sorted` existing.

---

## 4.2 Time series — how does it change?

**The question sounds like:** is it going up, is there a weekly rhythm, did something change
in March.

**The plots:** a line for the series itself; a rolling average alongside the raw values, never
instead of them; `seasonal_decompose` to split trend from seasonality from residual; an
autocorrelation plot to find a period you did not already suspect; a Fourier spectrum when
the period is long relative to your intuition.

**How it lies:**

- **Smoothing away the finding.** A rolling window wide enough to look clean is often wide
  enough to erase the event you were looking for. Plot raw and smoothed together. If they
  disagree about the thing you are claiming, the claim is about your window.
- **Gaps in the index plotted as straight lines.** A week with no data draws a confident
  diagonal between the weeks either side. Reindex to a complete date range and fill
  explicitly, so an absence looks like an absence. `pd.date_range` + `.reindex(fill_value=0)`
  — and think about whether zero is right, or whether missing means missing.
- **Confusing a trend with a cycle.** Six months of data cannot distinguish a rising trend
  from the upswing of an annual cycle. Neither can a decomposition; it will happily give you
  both components and no warning that you lacked the span to separate them.
- **The x-axis that lies about spacing.** Categorical x-ticks ("2021-W03") are evenly spaced
  regardless of the real time between them. A datetime axis is not. Know which one you have.
- **Aggregation window chosen after seeing the result.** Daily, weekly and monthly views of
  the same data support different stories. Choose the window from the mechanism you are
  studying, before you look.

**In GOAD:** `ShiftValues`, `DiffValues`, `RollingAvg`, `SelectDataRange`, `TimeFeatures` are
all pipeline steps, so the smoothing window is a named parameter you can change in one line
and rerun. `ComparePlot` puts raw and smoothed on the same axes; `VerticalDate` marks the
moment you claim something changed, with a label, on the plot. `DecomposePlot` wraps
`seasonal_decompose` into the four-panel observed/trend/seasonal/residual view; `ACFPlot`
wraps `acf` with its confidence band, for finding a period you did not already suspect.

---

## 4.3 Distributions — how is one variable spread?

**The question sounds like:** what is typical, how much variation is there, is that value
unusual.

**The plots:** a histogram to see the shape; an ECDF when you care about quantiles or want to
compare two samples without bin choices getting in the way; a qq-plot to judge a fit; the
fitted pdf drawn over the histogram to see where it agrees and where it does not.

**How it lies:**

- **Reporting a mean for a long-tailed variable.** The mean of a lognormal is not typical of
  anything; more than half the observations are below it. Message length, income, city size,
  response time, file size — all lognormal-ish, all routinely summarised with a mean and a
  standard deviation that describe no actual case.
- **Bin count as an unstated parameter.** The same data with 10, 30 and 100 bins can look
  unimodal, bimodal and noisy. If a claim depends on the bin count, it is a claim about the
  bin count.
- **"Outlier" as a synonym for "large".** A point three standard deviations out is
  extraordinary under a normal distribution and completely ordinary under an exponential one.
  Before removing anything, fit the distribution you actually believe. Most "outliers" in
  human-generated data are the tail doing exactly what a tail does.
- **Assuming normal because of the central limit theorem.** The CLT applies to *sums and
  means* of many independent contributions. It says nothing about a variable that is a
  product of factors (lognormal), a waiting time (exponential), a count of rare events
  (Poisson), or a rank-frequency (power law) — which between them cover most of what a
  human-behaviour dataset contains.
- **Comparing two groups by their histograms alone.** Overlapping histograms with different
  sample sizes are near-unreadable. Normalise, or use ECDFs, or do a two-sample test.

**In GOAD:** `HistogramPlot` (density-scaled, so a pdf overlays correctly), `DistPlot` for a
parametric curve, `DistributionFitter` + `PlotFits` for the full ranked comparison, and
`QQPlot` / `ECDFPlot` for the tail diagnostics a histogram cannot show. See
[Distributions](05-distributions.md).

---

## 4.4 Relations — do these two move together?

**The question sounds like:** does more of X mean more of Y, is the relationship linear, is
the correlation real.

**The plots:** a scatter first, always; a regression fit only after the scatter told you what
kind (`fit_reg` for linear, `lowess` for "let the data decide", `order=2` for an obvious
curve); a correlation matrix when there are many candidate pairs and you need to know where
to look.

**How it lies:**

- **Summary statistics without the picture.** Anscombe's quartet: four datasets, identical
  means, variances, correlations and regression lines, four completely different pictures.
  The Datasaurus dozen makes the same point with a dinosaur. Nothing else in data analysis is
  this cheap to demonstrate or this frequently ignored.
- **A correlation created by a third variable.** Both variables track time, or size, or
  activity level. The correlation is real and the implied mechanism is not. If two variables
  both grow, they correlate; that is arithmetic, not a finding.
- **A straight line through a curve.** `linregress` returns a slope and an r-value for
  anything you hand it, including a parabola, including a step function.
- **Overplotting.** Ten thousand points is a solid blob whose density you cannot read. Alpha,
  sampling, hexbin, or a 2D density — and note that the blob's shape is not its density.
- **Reading a correlation matrix as a ranking.** Correlations between correlated predictors
  are not independent evidence. This is exactly why regularisation is the better tool for
  "which variables matter" — see [Models and residuals](06-models-and-residuals.md).

**In GOAD:** `ScatterPlot`, `RegPlot` and `CorrelationHeatmap`, in that order of use.

```python
scatter = ScatterPlot(settings)
scatter.plot(data=data, x="messages", y="length", hue="is_bot")
scatter.plot_on(RegPlot(settings), data=data, x="messages", y="length", scatter=False)
```

The layering is the point: the scatter is drawn first and the fit goes on top of it, so the
choice of `fit_reg`, `order` or `lowess` is made after you have seen the shape rather than
before. `CorrelationHeatmap` is the entry point when there are too many pairs to plot — it
tells you which scatter to make next, and nothing more than that.

The interesting work is still in [Models and residuals](06-models-and-residuals.md), because
a relation you can state as a model is a relation whose failures you can inspect — and
§6.7 is how you check that the relation is not something a shuffle would have produced.

---

## 4.5 High-dimensional — what structure is in many variables at once?

**The question sounds like:** do these items form groups, which items are similar, what are
the main axes of variation.

**The plots:** a 2D projection (PCA for a linear one that preserves global structure, t-SNE or
UMAP for a nonlinear one that preserves neighbourhoods); a scree plot to say how much you
kept; a distance matrix when the items are few enough.

**How it lies:**

- **Reading t-SNE distances as distances.** t-SNE preserves who is near whom, not how far.
  Cluster sizes, the gaps between clusters, and the overall layout are artefacts. Two
  clusters far apart in a t-SNE plot are not more different than two that are close.
- **Perplexity as an undisclosed parameter.** The same data at perplexity 5, 30 and 100
  produces different numbers of apparent clusters. Sweep it. If the structure survives the
  sweep, it is structure; if not, it was the parameter.
- **PCA on unscaled features.** The component with the largest units wins. A column in
  milliseconds will dominate a column in years for reasons that have nothing to do with
  information.
- **A projection with no variance figure.** "Here are the first two components" means nothing
  without "which capture 34% of the variance". Put it in the title.
- **Curse of dimensionality.** In high dimensions everything is roughly equidistant from
  everything else, so nearest-neighbour intuitions quietly stop working — which is precisely
  the intuition a 2D projection restores to you, whether or not it is warranted.
- **Clustering the projection instead of the data.** Run the clustering in the original
  space; use the projection to *look* at the result. Clustering the 2D coordinates is
  clustering an artefact.

**In GOAD:** projections are computed elsewhere (sklearn) and plotted through a `BasePlot`
subclass, so the variance-explained ends up in the `PlotSettings` title where a reader will
see it.

---

## 4.6 Choosing between families

Some questions genuinely sit between two families, and the choice is the analysis:

- *"Do these two authors write differently?"* — a distributions question (compare two message
  length distributions), a categories question (mean length per author), or a
  high-dimensional one (embed the messages and see whether the authors separate). All three
  are valid; they answer different things and the first one is almost always the right place
  to start.
- *"Is activity increasing?"* — time series, unless the interesting part is *who* is driving
  it, which makes it categories over time.
- *"Is this value unusual?"* — distributions, always. Not "is it far from the mean" but "what
  fraction of the fitted distribution lies beyond it".

When stuck: the question you can most easily be wrong about is the one worth plotting first.

---

Next: [Distributions](05-distributions.md) — the family GOAD has the most machinery for.
