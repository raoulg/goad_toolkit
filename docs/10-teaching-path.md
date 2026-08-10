# 10. Teaching path

GOAD is the toolkit behind the MADS course **Data Analysis & Visualisation**
([MADS-DAV](https://github.com/raoulg/MADS-DAV)). This chapter maps the course onto these
chapters, and states what each lesson is actually trying to provoke.

The course premise: every student analyses their own exported chat history. The data differs
per student; the techniques do not. Each lesson therefore has two halves — a **showcase** on
a dataset chosen so the pattern is unmissable, and **your turn** on the student's own data,
where a null result is a legitimate, reportable outcome.

---

## Lesson 1 — Cleaning and enrichment

**Chapters:** [Pipelines](02-pipelines.md)

**The technique:** turning a raw export into an analysis-ready frame, as a `Pipeline` of
named transforms rather than forty notebook cells. Regular expressions as feature extraction.

**The question it provokes:** *what is one row, and what does this data not yet contain that
my question needs?* Most students arrive expecting cleaning to be janitorial. Enrichment is
where the analysis is won, and this is the lesson that says so.

**The exercise that matters:** write your own `TransformBase` subclass. Not because the
library needs it, but because the moment a student writes one they understand what
composition buys them — and every later lesson reuses the habit.

---

## Lesson 2 — Comparing categories

**Chapters:** [The five families](04-five-families.md) §4.1,
[Plot composition](03-plot-composition.md), [Visual critique](07-visual-critique.md)

**The technique:** bar, grouped bar, barbell, heatmap. Palettes, ordering, top-k.
`groupby`/`agg` and binning.

**The question it provokes:** *is there a variable that differs between these groups and also
affects the outcome?* Simpson's paradox belongs here, because it is a failure of categorical
comparison and it needs a picture to land.

**The trap to stage:** forty unordered bars in forty colours. Show it, then fix it, then name
which guideline each fix came from.

---

## Lesson 3 — Time series

**Chapters:** [The five families](04-five-families.md) §4.2, [Pipelines](02-pipelines.md)

**The technique:** timestamp features, reindexing to fill gaps, rolling averages, small
multiples per weekday, seasonal decomposition, autocorrelation, Fourier.

**The question it provokes:** *how much of what I see is the data, and how much is my
smoothing window?* Raw and smoothed on the same axes, always.

**The trap to stage:** a gap in the index drawn as a confident straight line. Continuity is a
gestalt principle, and here it lies for you.

**Own-data payoff:** message counts per day have a 7-day period that
`seasonal_decompose` finds unambiguously — one of the few own-data results that works for
essentially every student.

---

## Lesson 4 — Distributions

**Chapters:** [Distributions](05-distributions.md),
[The five families](04-five-families.md) §4.3

**The technique:** histograms, ECDFs, quantiles, qq-plots, the distribution registry,
`DistributionFitter`, KS versus log-likelihood, `PlotFits`.

**The question it provokes:** *is that value an outlier, or did I assume the wrong
distribution?* This is the most transferable idea in the course and it is invisible until a
student has fitted something other than a normal.

**Why chat data is unusually good here:** four variables, four different textbook families,
all from one file — messages per day (Poisson-ish, over-dispersed), seconds between messages
(exponential within bursts, heavy tailed overall), message length (lognormal), word rank
versus frequency (power law). A student can meet four generative mechanisms in one afternoon
on data they care about.

**The exercise that matters:** register a family GOAD does not ship — `pareto`, for the word
frequencies — and re-run the fit. Extension as a two-line act.

---

## Lesson 5 — Relations, models and residuals

**Chapters:** [Models and residuals](06-models-and-residuals.md),
[The five families](04-five-families.md) §4.4

**The technique:** scatter plots and their channels, linear/lowess/polynomial fits,
correlation matrices, regularisation as feature selection. Then the full loop: model, fit,
residual, distribution of the residual.

**The question it provokes:** *what did my model miss, and when?* The covid example makes
this concrete — the vaccination effect is visible in the residual and invisible in the raw
series.

**The argument to lead with:** Anscombe's quartet and the Datasaurus dozen. Identical summary
statistics, different pictures. Nothing else in the course argues "plot it" this efficiently.

**The framing that outlasts the techniques:** the map is not the territory. A model is a
compression, and the question is always what it discarded, not whether it is true.

---

## Lesson 6 — High-dimensional data

**Chapters:** [The five families](04-five-families.md) §4.5

**The technique:** count vectors and n-grams, then embeddings; PCA and t-SNE; distance
metrics; the curse of dimensionality.

**The question it provokes:** *how much of this cluster structure is in the data, and how much
is in the projection?* Sweep the perplexity. If the structure survives, it is structure.

**The comparison worth making:** a hand-built count-vector representation against a
sentence-embedding one, on the same task — can you separate the authors? The cheap
interpretable one wins often enough to be worth the demonstration, and that result is the
best argument in the course for trying the simple thing first.

---

## Lesson 7 — Dashboards

**The technique:** streamlit, session state, layout, wiring a model into an app.

**The question it provokes:** *what does a reader need to be able to change, and what should
you have decided for them?* An interactive plot is not automatically a better plot; every
control is a decision you declined to make.

**The payoff:** a `BasePlot` subclass written in lesson 2 renders inside the dashboard in
lesson 7 unchanged. That is what the composition discipline was for.

---

## Cross-cutting threads

**1. The residual is the finding.** Introduced in lesson 1 as "what does the data not
contain", made literal in lesson 5, and the standing answer to "I made a plot, now what".

**2. Hand-designed versus learned features.** Regex features in lesson 1, count vectors in
lesson 6, embeddings at the end of lesson 6. Moving a decision across that line — and
noticing that the hand-designed side often wins on small data — is the thread that continues
into the machine learning course.

**3. Every parameter is part of the claim.** The bin count, the smoothing window, the top-k
cut, the perplexity. A finding that does not survive a sweep of its parameter is a finding
about the parameter.

**4. Grey first.** Introduced in lesson 1's critique checklist, applied in every lesson,
graded in every submission.

---

## Working with an assistant, as a course policy

Students will use assistants. The question is whether they use them as a way to skip the
reasoning or as a way to do more of it.

The `goad` MCP server exists for the second case: it hands the assistant the questions from
[The analysis method](09-analysis-method.md) and [Visual critique](07-visual-critique.md) and
instructs it to ask the student, one stage at a time, rather than answer them. See
[MCP_SERVER.md](../MCP_SERVER.md).

The assessment consequence, stated plainly to students: you may use any tool you like, and
you will be asked to defend a choice without it. An analysis you cannot explain is an
analysis you did not do.
