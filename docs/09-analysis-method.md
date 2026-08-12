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
| Did this question come from an idea about the world, or from a pattern you already spotted in the data? | Both are legitimate — exploration is how most real questions are found. Confirming a data-suggested pattern on the data that suggested it is not | which of the two; and if from the data, the later period, other group, or second measure it will be checked against |
| If the answer is yes, what mechanism would explain it — and would you have predicted it? | The same result means something different for "fewer messages at 4am" (a mechanism anyone can name) than for "the 4th floor is grumpier" (none) | the mechanism, or an explicit "none — so a positive result sends me looking for a confounder" |
| Who reads the answer, and how long do they have? | Decides how much must be visible in the picture versus how much can stay in your head | an audience and a time budget |
| Does anyone care what the answer is? | If both answers are equally uninteresting, find a better question now, not after a week of work | a consequence that differs by outcome |

**The most common failure:** starting from a dataset instead of a question. "I have chat data,
what can I find" produces a tour of the columns. Turn it around: what do you actually wonder
about these conversations?

**On the third and fourth rows.** Finding a question by exploring is not the failure — it is
how most real questions are found, and this whole method is built for data you have not seen
before. The failure is letting the same data both suggest a pattern and confirm it. Naming
which kind of question you have decides what would count as evidence, and it is free to do
now and impossible to reconstruct later.

The fourth row is the one that answers a case statistics cannot. If a difference between
floors of an office building comes out significant, the right response is not "we found an
effect" — it is "what mechanism could possibly produce that?", and if the honest answer is
none, then what you have found is a fluke or a confounder worth chasing (perhaps the 4th
floor is one department). The same evidence should move you differently depending on whether
a mechanism exists. See §9.6.

---

## 9.2 Stage 2 — The data

**Settle:** what one row is, what was measured versus derived, and what enrichment the
question needs.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| What is one row? | If you cannot say it in a sentence, later aggregations will be wrong in ways that do not error | the unit of observation |
| What is the unit your **claim** is about, and is it the same as one row? | 4000 messages from 5 people is n=5 for a claim about people. Rows that cluster inside a person are one observation measured many times | the claim's unit, the row's unit, and whether they match |
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
| How many **independent units** sit behind each group you intend to compare? | Three people produce no summary stable enough to compare, however many rows they generated between them. And every set of numbers has a maximum, so one group is always "highest" | the count per group, at the claim's unit — not the row count |

**Do this before the interesting part, not after.** A correlation between two long-tailed
variables is usually a correlation between their two largest observations, and you will not
see that from the scatter plot alone.

**Pseudoreplication** is the name for getting the last row wrong, and it is worth knowing,
because a named error is one you can catch in your own draft. It looks like this: a bar chart
of message length by gender, thousands of rows behind each bar, an overwhelming difference —
built from three men and two women. The bar chart has n=4000. The claim has n=5. Aggregate to
one number per person and plot those five points instead; the groups will usually overlap
completely. Same data, two units of analysis, opposite conclusions.

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
| Is the finding written on the plot? (`Annotate`) | A reader with thirty seconds reads the annotation, not the axes | the annotation, and where it sits |

See [Visual critique](07-visual-critique.md).

---

## 9.6 Stage 6 — The verification

**Settle:** how you would know the pattern is real, before you write it down as a finding.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| What model of "nothing interesting is happening" would produce this too? | This is what a claim is measured against | the null you are ruling out |
| How many comparisons did you look at before this one — including the abandoned ones? | With fifteen things you could have measured, the chance at least one looks striking by luck alone is better than even | how many were considered, and whether the finding was confirmed on data that did not suggest it |
| If you shuffled the labels and re-measured, how often would you see this? | The cheapest honest check available, and it needs no theory | a resampling procedure and a rough frequency |
| Is there a third variable that could produce this pattern? | The most common way a real correlation implies a false mechanism | the confounder, and how you checked it |
| Does the pattern survive a slice you did not use to find it? | A pattern found and confirmed on the same data is not confirmed | the held-out slice |
| Does it survive changing the parameter from stage 4? | If it does not, the finding is about the parameter | the sweep you ran |
| If you modelled it, what does the residual look like? | Structure in the residual is an unmodelled mechanism, not noise | the residual's shape over time and its distribution |

See [Models and residuals](06-models-and-residuals.md) §6.3 for the residual loop, and §6.7
for the shuffle test — `NullDistribution` runs the resampling once you have said what to
shuffle.

### The garden of forking paths

The second row deserves its own paragraph, because it is the failure that survives every
other check on this list.

Suppose there is nothing to find. You measure fifteen plausible things anyway — message
length, posting hour, question rate, emoji rate, response time, weekend share, and so on —
and report whichever looked most striking. At the conventional one-in-twenty threshold, the
chance that at least one of fifteen independent measurements clears it by luck is
`1 - 0.95^15 ≈ 0.54`. More likely than not, on data with no signal in it whatsoever.

Nothing about the winning comparison looks wrong. It has a real difference, a real plot, and
a real number. What is missing is the fourteen you did not report, and no amount of scrutiny
applied to the survivor can recover them.

Three defences, in order of strength:

1. **Write the question down before you look** (§9.1). Then there is one comparison, and its
   result means what it says.
2. **Split the data.** Hunt on one half, confirm on the other. A finding that does not
   survive the second half was never a finding.
3. **Say how many you looked at.** "The largest of fifteen differences we examined" is an
   honest and perfectly reportable sentence. "We found that…" is not the same sentence.

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
