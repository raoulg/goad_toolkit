# 9. The analysis method

The ordered version of [Goal oriented analysis](01-goal-oriented-analysis.md). Six stages,
in the order they must happen, because each one is unanswerable until the previous is
settled.

The `goad_analysis_checklist` MCP tool walks these one at a time and will not advance until
you have answered the current stage in your own words. This chapter is the same content in
one place, for reading.

---

## 9.1 Stage 1 — The question

**Settle before anything else:** what you want to know, what would count as an answer in both
directions, and who the answer is for.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| What do you want to know, in one sentence? | Every later decision is downstream of this, and this is the stage people skip | a proposition, not a topic |
| What result would make you say "no"? | An unfalsifiable question cannot be answered, only decorated | the specific pattern whose absence settles it |
| Who reads the answer, and how long do they have? | Decides how much must be visible in the picture versus how much can stay in your head | an audience and a time budget |
| Does anyone care what the answer is? | If both answers are equally uninteresting, find a better question now, not after a week of work | a consequence that differs by outcome |

**The most common failure:** starting from a dataset instead of a question. "I have chat data,
what can I find" produces a tour of the columns. Turn it around: what do you actually wonder
about these conversations?

---

## 9.2 Stage 2 — The data

**Settle:** what one row is, what was measured versus derived, and what enrichment the
question needs.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| What is one row? | If you cannot say it in a sentence, later aggregations will be wrong in ways that do not error | the unit of observation |
| Which columns were measured, which were derived, and by whom? | A derived column carries somebody's assumptions, including yours from an hour ago | the provenance of each column you rely on |
| What does your question need that the data does not contain? | This is the enrichment step, and it is usually where the analysis is won | the specific derived features |
| What is missing, and is missing informative? | A silent gap is a data point about the process | the pattern of missingness |
| Which of this belongs in a pipeline? | Anything you would run again on new data | the steps, named |

**Enrichment is undervalued.** A hand-written regex feature encoding something you know about
the domain regularly beats a more sophisticated model on the same raw columns, is
interpretable, and takes four minutes. Reach for it before reaching for a bigger method.

See [Pipelines](02-pipelines.md).

---

## 9.3 Stage 3 — The shape

**Settle:** for each variable in play, what its distribution looks like, before you compare,
correlate or model anything with it.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| Discrete or continuous? | Selects which half of the distribution registry can even apply | which, per variable |
| What does the histogram look like — symmetric, skewed, bounded, multimodal? | A mean summarises only one of those honestly | the shape, in words |
| Is it long-tailed? | Long tails break means, standard deviations, and every standard outlier rule | yes/no, and what you will report instead |
| Are the extreme values errors, or ordinary members of a tail? | Removing tail members because they are large is the most common way to destroy a finding | the criterion separating the two |
| Is it multimodal — and if so, which hidden variable are you looking at? | Two peaks usually means two populations mixed | the candidate grouping variable |

**Do this before the interesting part, not after.** A correlation between two long-tailed
variables is usually a correlation between their two largest observations, and you will not
see that from the scatter plot alone.

See [Distributions](05-distributions.md).

---

## 9.4 Stage 4 — The encoding

**Settle:** which family the question lives in, and which visual channel carries the
comparison you actually care about.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| Which of the five families is this question in? | The family follows from the question; choosing a chart type first is choosing an answer first | categories / time / distributions / relations / high-dimensional |
| Which comparison is the point? | The point gets position or length; everything else gets a weaker channel | the one comparison |
| What is on each axis, and what does the spacing mean? | Categorical axes are evenly spaced regardless of real distance | both axes, with units |
| What aggregation happened, and could it hide the finding? | A mean over a bimodal group describes nobody | the aggregation, and what it discards |
| Which parameter, if changed, would change the conclusion? | Bin count, smoothing window, perplexity, top-k cut — each is part of the claim | the parameter, and the range you checked |

See [The five families](04-five-families.md).

---

## 9.5 Stage 5 — The critique

**Settle:** whether the picture makes the argument, judged the way a reader will judge it.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| Look away and back — what did you see first? | The preattentive channels carry the message whether or not you chose them | what caught your eye, and whether it should have |
| What is grouped visually, and does it match what is grouped conceptually? | The gestalt principles group things for the reader without asking you | the groupings, intended and accidental |
| Which single element carries the message, and is it the only coloured thing? | Grey first, colour as a pointer, not decoration | the one element |
| What can you delete without losing meaning? | Everything remaining should be earning its place | the list you deleted |
| Is the finding written on the plot? | A reader with thirty seconds reads the annotation, not the axes | the annotation, and where it sits |

See [Visual critique](07-visual-critique.md).

---

## 9.6 Stage 6 — The verification

**Settle:** how you would know the pattern is real, before you write it down as a finding.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| What model of "nothing interesting is happening" would produce this too? | This is what a claim is measured against | the null you are ruling out |
| If you shuffled the labels and re-measured, how often would you see this? | The cheapest honest check available, and it needs no theory | a resampling procedure and a rough frequency |
| Is there a third variable that could produce this pattern? | The most common way a real correlation implies a false mechanism | the confounder, and how you checked it |
| Does the pattern survive a slice you did not use to find it? | A pattern found and confirmed on the same data is not confirmed | the held-out slice |
| Does it survive changing the parameter from stage 4? | If it does not, the finding is about the parameter | the sweep you ran |
| If you modelled it, what does the residual look like? | Structure in the residual is an unmodelled mechanism, not noise | the residual's shape over time and its distribution |

See [Models and residuals](06-models-and-residuals.md) §6.3.

---

## 9.7 Then go round again

The output of stage 6 is rarely "yes, done". It is usually a sharper version of stage 1 — the
residual has structure in a particular period, or the pattern held for one group and not
another. That is not a failed analysis. That is the analysis, and the second lap is where
findings come from.

Stop when the residual is boring and you can state the claim, its scope, and what would
falsify it, in three sentences.

---

## 9.8 Working with an assistant

The division that keeps this method intact:

**Yours, always:**

- the question and what would count as an answer (stage 1)
- what one row means, and which enrichment is worth building (stage 2)
- whether an extreme value is an error or data (stage 3)
- what you are claiming (stage 5, §7.4)
- whether a residual's structure corresponds to something real in the world (stage 6)

**Reasonable to delegate:**

- writing the transform once you have named it
- remembering the seaborn or scipy argument
- catching a rainbow palette, an unsorted category axis, a legend that could be direct labels
- generating the resampling loop for stage 6, once you have said what to shuffle

**The failure mode to watch for:** an assistant given the whole method at once will answer all
six stages itself, quickly and plausibly, and you will end up defending conclusions you did
not reach. That is why `goad_analysis_checklist` returns one stage at a time and records
*your* answer before advancing. If your assistant has run ahead, ask it to go back to stage 1
and actually wait for you.

---

Next: [Teaching path](10-teaching-path.md).
