# 5. Distributions

## 5.1 Why fit a distribution at all

Three reasons, in increasing order of importance:

1. **To summarise honestly.** "Lognormal with μ=3.2, σ=0.8" describes the data. "Mean 34,
   std 41" describes nothing, because for a lognormal the mean is not typical and the std
   implies a symmetry that is not there.
2. **To decide what is unusual.** An outlier is only an outlier relative to an assumed
   distribution. Without a fit, "unusual" means "far from the mean", which is a statement
   about the normal distribution you did not know you had assumed.
3. **To learn the mechanism.** Distribution families come from processes. Counts of
   independent rare events give Poisson. Waiting times between them give exponential.
   Products of many factors give lognormal. Sums of many give normal. Preferential attachment
   gives a power law. When your data fits a family, you have a hypothesis about *how it was
   generated* — and that is a finding, not a summary.

Point three is why this chapter exists. Fitting is not curve-matching; it is asking what
kind of process you are looking at.

## 5.2 The registry

`DistributionRegistry` holds the families GOAD will try, each with the metadata the fitter
needs: the scipy object, whether it is discrete, and how many parameters it takes.

Shipped: `norm`, `uniform`, `lognorm`, `poisson`, `exponential`, `skewnorm`, `gamma`,
`weibull`.

```python
from goad_toolkit.distributions import DistributionRegistry
from scipy import stats

registry = DistributionRegistry()
registry.get_names()
registry.register_distribution("pareto", stats.pareto, is_discrete=False, num_params=3)
```

Registering a family is the extension point that matters most in practice, because the
shipped set is missing things real data needs — `pareto` and `zipf` for rank-frequency data,
`bernoulli` and `binom` for yes/no outcomes, `nbinom` for over-dispersed counts, `beta` for a
distribution over a probability.

Two things to know before you rely on it:

- **`num_params` must be right.** It selects the bounds template the fitter builds (see §5.3),
  so a wrong count produces a confusing optimisation failure rather than a clear error.
- **Registrations are scoped to the registry object.** Every `DistributionRegistry()` starts
  from the shipped families, so a registration in one cell affects only the fits that use that
  registry, and a fresh instance is your clean slate — no kernel restart needed. Pass the
  registry you extended to the fitter:

  ```python
  fitter = DistributionFitter(registry)
  ```

  `DistributionFitter()` without an argument builds its own registry with the shipped
  families.

## 5.3 `DistributionFitter`

```python
from goad_toolkit.analytics import DistributionFitter

fitter = DistributionFitter()
results = fitter.fit(data["residual"], discrete=False)
best = fitter.best(results, criterion="combined")
```

`fit` walks the registry, keeps only families matching the `discrete` flag, and fits each one
by maximum likelihood. For each family it returns either a `FitResult` or a `FailedFit` —
failures are values, not exceptions, so one badly-behaved family does not lose you the rest
of the comparison.

A `FitResult` carries:

| Field | What it is |
|---|---|
| `distribution` | the family name |
| `params` | the fitted parameters |
| `frozen_dist` | a scipy distribution with those parameters applied — ready to plot or sample |
| `log_likelihood` | sum of log-pdf over the data |
| `kstest` | a `KSTestResult` with `statistic` and `p_value` |
| `best_likelihood` / `best_ks` | flags marking the winner under each criterion |

**Bounds.** `stats.fit` needs bounds, and the fitter derives them from the data: location
bounded by the data range extended either side, scale from a fraction of the standard
deviation up to twenty times it, shape parameters in `(0.01, 10)`. These are generous
defaults, not truths. A family whose real shape parameter is above 10 cannot be fitted here,
and the symptom is a fit that pins to the boundary rather than an error. If a fit looks
suspiciously like a boundary value, that is why.

**`discrete` is your call, not the fitter's.** Counts (messages per day) are discrete;
durations (seconds between messages) are continuous. Getting it wrong does not error — it
silently fits the wrong half of the registry.

## 5.4 KS versus log-likelihood

The fitter marks two winners because the two criteria measure different things and
disagreeing is normal.

**Log-likelihood** rewards a family that assigns high probability to the points you observed.
It is dominated by the bulk of the data, and it punishes severely any family that puts near-
zero probability where you have observations. It cannot be compared across datasets, only
across families on the *same* data — and it slightly favours families with more parameters,
which is what AIC/BIC corrections exist to offset.

**The Kolmogorov–Smirnov test** measures the largest gap between the empirical and the
fitted cumulative distributions, and reports the probability of seeing a gap that large if
the fit were correct. It is sensitive to the middle of the distribution and comparatively
blind in the tails.

So:

- **High log-likelihood, poor KS p-value** — the family fits the bulk well but the shape is
  systematically off somewhere. Look at the qq-plot; the deviation is usually in a tail.
- **Good KS p-value, lower log-likelihood** — the overall shape is plausible but the family
  spreads probability thinly. Common for heavy-tailed families on small samples.
- **Both agree** — as much confidence as this method offers.

Two warnings about the p-value specifically. First, it is *not* the probability that the
family is correct; it is the probability of this much deviation given that it is. Second, KS
p-values are optimistic when the parameters were estimated from the same data — which is
exactly what happened here. Treat it as a ranking device between families, not as a
hypothesis test you would report.

And the standing rule: with enough data, every fit is rejected, because no real dataset is
exactly anything. With little data, nothing is rejected. Neither situation makes the picture
unnecessary.

## 5.5 Reading `PlotFits`

```python
from goad_toolkit.visualizer import PlotFits, PlotSettings, FitPlotSettings

settings = PlotSettings(figsize=(12, 6), title="Residuals",
                        xlabel="error", ylabel="probability")
fig = PlotFits(settings).plot(
    data=data["residual"],
    fit_results=results,
    fitplotsettings=FitPlotSettings(bins=30, max_fits=3),
)
```

One panel per family, ranked by log-likelihood, each showing the data histogram in grey with
the fitted pdf over it. The panel title carries the KS p-value and the log-likelihood; the
best-by-likelihood fit is drawn in crimson, the best-by-KS in dark blue, everything else in
grey. When one panel is both colours' winner, the criteria agreed.

`max_fits` limits how many panels you get. Three is usually right — you are comparing, not
cataloguing.

What to look at, in order:

1. **The tails.** This is where families differ and where the fit matters, and it is also
   where a histogram has the fewest observations and is least readable. If your question is
   about extreme values, a histogram will not settle it — use a qq-plot or an ECDF on a log
   scale.
2. **Systematic deviation, not point-by-point wobble.** A curve that is above the histogram
   on the left and below it on the right is a wrong-shape signal. Individual bars being off
   is sampling noise.
3. **Whether the winner is meaningfully ahead.** Two families with near-identical
   log-likelihoods have not been distinguished by your data, and picking one is a choice you
   should make on mechanism, not on the fourth decimal place.

## 5.6 Outliers, properly

The sequence that avoids the usual mistake:

1. **Look at the distribution first.** Not the summary statistics.
2. **Fit before you judge.** A point at the 99.9th percentile of a fitted exponential is
   expected in a sample of a few thousand. The same point is a five-sigma event under a
   normal fit. Same data, opposite conclusions, and the difference is which family you
   assumed.
3. **Separate errors from tails.** An error is a value that could not have been produced by
   the process — a negative duration, a timestamp in 1970, a message length of 40 million
   characters. A tail value is unusual but possible. The first should be removed and
   documented; the second is data.
4. **If you remove anything, report it.** How many, on what criterion, and what the analysis
   looks like with them left in. An analysis that changes conclusion when three points are
   removed is a finding about those three points.
5. **Consider that the outlier is the point.** In human-behaviour data the extreme values are
   often the interesting cases, not the contamination.

The trap worth naming: standard outlier rules — z-score above 3, or 1.5×IQR beyond the
quartiles — encode a normality assumption. Applied to a lognormal or an exponential, they
flag a fixed fraction of perfectly ordinary observations, every time, no matter how clean the
data is.

## 5.7 What the families mean

| Family | Generated by | You will meet it as |
|---|---|---|
| normal | sums of many independent contributions | measurement error, aggregates, model residuals |
| lognormal | *products* of many factors | message length, income, response time, file size |
| exponential | waiting time between independent events | time between messages, within a burst |
| Poisson | counts of independent rare events in a fixed window | messages per hour, arrivals per day |
| power law / Pareto | preferential attachment, "rich get richer" | word frequency by rank, followers, city size |
| gamma / Weibull | sums of exponentials; time-to-failure with changing hazard | durations, lifetimes |
| beta | a distribution *over a probability* | uncertainty about a rate, A/B test posteriors |
| uniform | a bounded process with no preference | rarely, in nature; often, in bugs |

Two practical notes. **Over-dispersion**: real counts are usually more variable than Poisson
predicts, because the rate itself varies (people are more active on some days). When Poisson
fits badly but the shape is right, a negative binomial is the standard next step. And
**lognormal versus power law** is genuinely hard to distinguish from a histogram — both look
like "long tail". The rank-frequency plot on log-log axes is the discriminator: a power law
is straight there, a lognormal curves.

## 5.8 Fitting the residual

The highest-value application, and the one `demo/linear.py` demonstrates: fit a model, take
what it did not explain, and fit a distribution to *that*.

- **A clean symmetric residual from a plausible family** means the model extracted the
  structure it could. What is left is noise, and you should stop.
- **A skewed residual** means the model is systematically wrong in one direction — often a
  sign that the relationship is multiplicative and you fitted it additively.
- **A heavy-tailed residual** means occasional large misses. Something the model does not
  know about happens sometimes. Find out when.
- **A residual with structure over time** — a step, a drift, a periodicity — is not a
  distribution problem at all. It is an unmodelled mechanism, and locating *when* it starts
  is usually more informative than any summary of it.

That last case is the one that turns an analysis into a finding. See
[Models and residuals](06-models-and-residuals.md).

---

Next: [Models and residuals](06-models-and-residuals.md).
