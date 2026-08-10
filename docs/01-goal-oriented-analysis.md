# 1. Goal oriented analysis

## 1.1 The failure this method is built against

Open a dataset. Call `.describe()`. Plot a histogram of every numeric column. Make a
correlation heatmap. Notice that two variables correlate. Write that down.

Nothing in that sequence is wrong, and the whole thing is useless. It produces output
proportional to the number of columns and insight proportional to nothing. The reason is
structural: no step in it can fail. There is no result that would have made you stop, back
up, or change your mind, because there was never a question that a result could disagree
with.

Goal Oriented Analysis of Data is the opposite discipline. Every step is chosen because a
specific question needs it, and every step can come back with an answer you did not want.

## 1.2 The question comes first, and it has a shape

A usable question names three things:

1. **What you want to know.** Not a topic — a proposition. "Whether messages cluster in the
   evening" rather than "message timing".
2. **What would count as an answer.** Both directions. If you cannot describe the result
   that would make you say "no, they don't", you do not yet have a question.
3. **Who the answer is for.** This decides how much has to be visible in the picture versus
   how much can live in your head. A plot for yourself and a plot for a reader who has
   thirty seconds are different objects.

Bad: "Let's explore the chat data."
Better: "Is there a daily rhythm in when messages are sent?"
Good: "Are messages concentrated in two daily peaks, and is the evening peak larger than the
morning one — enough that a reader would see it without being told?"

The third version is worth writing down because it can be wrong. It also tells you what plot
to make, which the first version never will.

## 1.3 The loop

```
question -> enrich -> look -> model -> residual -> fit -> (new question)
```

**Enrich.** Raw data rarely contains the variable your question is about. A chat export has
a timestamp; your question is about hour-of-day. It has message text; your question is about
whether a message is a question. This is where regex features, timestamp features, joins and
derived columns happen — and where a `Pipeline` earns its keep, because the enrichment is
the part you will run again and again on new data.
See [Pipelines](02-pipelines.md).

**Look.** Make the plot that most directly shows the thing you asked about, before making
any others. Which family of plot that is depends only on the question — see
[The five families](04-five-families.md).

**Model.** State what you expect, as a function or a distribution. This is the step people
skip, and skipping it is what makes an analysis unfalsifiable. A model can be trivially
simple: "messages per day is constant", "message length is lognormal", "deaths are
proportional to positive tests fourteen days earlier". Simple is fine. Stated is mandatory.

**Residual.** Subtract what you expected from what you saw. This is the actual finding. A
model that fits perfectly tells you nothing you did not already believe; a model that misses
tells you exactly where your understanding runs out, and *when* it runs out, which is usually
the interesting part.

**Fit.** Ask what the residual is made of. If it is unstructured noise from a plausible
distribution, the model has extracted everything there was. If the residual is skewed,
heavy-tailed, autocorrelated, or has a step in it, there is another mechanism you have not
named yet — and you now know something about its shape.
See [Distributions](05-distributions.md).

Then back to the top with a sharper question.

## 1.4 The residual is the finding

This is the load-bearing idea, so it gets its own section.

In `demo/linear.py`, a linear model predicts covid deaths from positive tests shifted by
fourteen days. The fit is decent. Nobody learns anything from the fit — of course deaths
track cases.

The residual is where the analysis actually happens. Plotted over time, it is not flat: it
drifts systematically after a particular date. That date is when vaccination started. The
model's *failure* located a change in the underlying mechanism, and did so more precisely
than staring at the raw series ever would, because the raw series is dominated by the part
the model already explains.

Generalised: a model is a hypothesis about what is boring. Subtracting it leaves the part
that is not boring. If you never build a model, everything stays mixed together and the
large boring component hides the small interesting one.

## 1.5 What "extendable" is for

Every one of GOAD's three subsystems is a base class plus a handful of implementations:

| Subsystem | Base class | Ships with | You add |
|---|---|---|---|
| Transforms | `TransformBase` | shift, diff, select-range, rolling average, z-score | whatever your data needs |
| Plots | `BasePlot` | line, compare, histogram, distribution, residual, fits | your chart, your annotation layer |
| Distributions | `DistributionRegistry` | norm, uniform, lognorm, poisson, expon, skewnorm, gamma, weibull | the family your data actually follows |

This is not a gap in the library that a future release will fill. It is the design. The set
of transforms any real analysis needs is unbounded and mostly domain-specific; the set worth
shipping is small and generic. What ships is a *worked example of the shape*, so that adding
yours is a subclass with one method, and it then composes with everything else for free.

The practical consequence: when GOAD does not do what you need, the correct move is almost
never to abandon it for raw pandas and matplotlib. It is to write the fifteen-line subclass
and keep the composition.

## 1.6 Where the method resists being handed over

An assistant that is given "analyse this dataset" will produce the `.describe()`-and-heatmap
output from §1.1, competently and instantly. That is the failure mode this method exists to
prevent, arriving faster.

The steps an assistant genuinely cannot do for you:

- **The question** (§1.2). It does not know what you want to know or who is going to read it.
- **The model** (§1.3). Stating what you expect is a claim about the world, and it is yours.
- **Deciding whether a residual is interesting.** It can tell you the residual is skewed. It
  cannot tell you that the skew corresponds to the week the group chat gained three members.

The steps it is genuinely good at: writing the transform once you have named it, remembering
the seaborn argument, catching that you plotted a proportion across groups of very different
size. Use it there.

The `goad_analysis_checklist` MCP tool exists to keep that division honest — it hands the
assistant the questions from §1.2 through §1.4 and instructs it to ask you rather than answer
them. See [The analysis method](09-analysis-method.md).

---

Next: [Pipelines](02-pipelines.md) — the enrich step, made reusable.
