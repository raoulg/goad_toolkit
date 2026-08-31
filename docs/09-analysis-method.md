# 9. The analysis method

The ordered version of [Goal oriented analysis](01-goal-oriented-analysis.md). Six stages,
in the order they must happen, because each one is unanswerable until the previous is
settled.

The `goad_analysis_checklist` MCP tool walks these one at a time and will not advance until
you have answered the current stage in your own words. This chapter is the same content in
one place, for reading.

---

## 9.1 Stage 1 — The question

**Settle before anything else, and get there by interview, not interrogation:** what this
data is, what you already knew before a single plot, what would count as an answer in both
directions, and who it's for. "What do you want to know" is answered with "I don't know" more
often than it is answered at all — "what have you half-noticed" gets further, faster.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| What is this data, and who or what is it a record of? | Grounds the conversation in what you already know before a column name gets to suggest a question for you | the group, subject, or process, and roughly what period it covers |
| What do you already know happened in here, before looking at a single number? | Whoever was there carries context no column will ever contain — and it's context an assistant reading only the rows cannot supply either, because it was not there | specific remembered events, roughly dated |
| What are the dynamics, running jokes, or character traits a stranger reading the raw rows would never guess? | This is where a real theme usually comes from — the ebb and flow of attention, who drives a conversation, what's the group's own | a named theme or dynamic |
| What's curious to you, and what do you suspect is really going on that you couldn't show just by scrolling a few messages? | The actual target of this stage — say it even unpolished | the suspicion, in your own words |
| Name a result that would be true and completely uninteresting even if you found it — is the question forming so far that one in disguise? | Every dataset has open doors ("a timeseries of message count shows people sleep at night"). Naming one on purpose is the fastest way past it | the boring result, and how the real question differs |
| Turn the most interesting thread above into one sentence a result could contradict — and what result would make you say "no"? | A suspicion is not yet a question; a question is a proposition with a way to lose | a proposition, not a topic, plus the pattern whose absence settles it |
| Did this come from something you already knew, or from a pattern you noticed while skimming the data? | Both are legitimate — exploration is how most real questions are found. Confirming a data-suggested pattern on the data that suggested it is not | which of the two; and if from the data, the later period, other group, or second measure it will be checked against |
| If this turns out true, what's the arc — the build-up, the turn, the punchline — and would you have called the punchline beforehand? | The same result means something different for "fewer messages at 4am" (an arc anyone can narrate) than for "the 4th floor is grumpier" (none) | the arc, or an explicit "none — so a positive result sends me looking for a confounder" |
| Who reads this, how long do they have, and would it actually surprise them? | Decides how much must be visible in the picture versus how much can stay in your head — and whether either outcome is worth a week of work | an audience, a time budget, a consequence that differs by outcome |

**The most common failure:** starting from a dataset instead of a question. "I have chat data,
what can I find" produces a tour of the columns. Turn it around: what do you actually wonder
about these conversations — and let that answer come from what you already know about the
group, not from staring at rows waiting for a question to appear in them.

**On the seventh row.** Finding a question by exploring is not the failure — it is how most
real questions are found, and this whole method is built for data you have not seen before.
The failure is letting the same data both suggest a pattern and confirm it. Naming which kind
of question you have decides what would count as evidence, and it is free to do now and
impossible to reconstruct later.

The eighth row is the one that answers a case statistics cannot. If a difference between
floors of an office building comes out significant, the right response is not "we found an
effect" — it is "what arc could possibly produce that?", and if the honest answer is none,
then what you have found is a fluke or a confounder worth chasing (perhaps the 4th floor is
one department). The same evidence should move you differently depending on whether a
mechanism exists. See §9.6.

---

## 9.2 Stage 2 — The data

**Settle:** which feature answers each theme stage 1 surfaced, what one row is, what was
measured versus derived, and what enrichment the question still needs.

Unlike stage 1, this is a stage worth working through together rather than one you sit back
and interview for. Turning a feeling into a feature is a technical translation, and an
assistant is genuinely useful here — proposing candidates, not just asking the question and
waiting for a finished answer. The student still decides what stays.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| For every theme, event, and suspicion named in stage 1, talked through together — does a feature come to mind, or is it worth brainstorming one? | A suspicion that never gets mapped onto a feature stays a feeling — it does not become part of the analysis just because it was said out loud | one feature per named theme, agreed between student and assistant — or an explicit "no feature for this one" |
| What is one row? | If you cannot say it in a sentence, later aggregations will be wrong in ways that do not error | the unit of observation |
| What is the unit your **claim** is about, and is it the same as one row? | 4000 messages from 5 people is n=5 for a claim about people. Rows that cluster inside a person are one observation measured many times | the claim's unit, the row's unit, and whether they match |
| Which columns were measured, which were derived, and by whom? | A derived column carries somebody's assumptions, including yours from an hour ago | the provenance of each column you rely on |
| Beyond those named themes, what does the stage-1 proposition still need that the data doesn't contain? | The formal proposition usually needs a few mechanical features the interview would not surface — an hour-of-day column, a rolling window, a join | the specific derived features |
| What is missing, and is missing informative? | A silent gap is a data point about the process | the pattern of missingness |
| Which of this belongs in a pipeline? | Anything you would run again on new data | the steps, named |

**Enrichment is undervalued.** A hand-written regex feature encoding something you know about
the domain regularly beats a more sophisticated model on the same raw columns, is
interpretable, and takes four minutes. Reach for it before reaching for a bigger method.

**On the first row.** Stage 1 is an interview precisely so themes surface before any column
is named. That only pays off if each one lands on an actual feature here — otherwise the
interesting part of stage 1 evaporates and stage 2 quietly turns back into a tour of the
columns.

See [Pipelines](02-pipelines.md).

---

## 9.3 Stage 3 — The shape

**Settle, first, whether this stage is needed at all — in plain language.** Most students
here have not met anything past a normal distribution, and leading with "shape",
"discrete", or "continuous" loses them before the question lands. Ask about common versus
rare, typical versus extreme, bucket versus number line instead, and let the technical
vocabulary arrive after the idea has, if at all.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| Is there anywhere in the claim where knowing something is common versus rare, or typical versus extreme, would actually change what you'd say? | The plain-language gate. A check that would not change the answer is not worth running — fitting a distribution to every column because the data has that many is the describe()-and-heatmap tour in a fancier outfit | the specific feature and where it bites — or an honest "nothing here would change, skip it" |
| For that one: if it turned out common/typical almost everywhere versus rare/wildly uneven, what would you actually do differently? | Naming the decision in plain terms turns "let's look at the histogram" from curiosity into a step with a point | the concrete decision, as an if/then, in the student's own words |
| Do you want to treat that feature as a small number of buckets (categories) or as a number line — and which serves the claim better? | Almost anything can be sliced into buckets or left continuous — this is a decision made for the claim, not a fact the column hands you, and the two choices support different plots | which was chosen, and why — not just "it's numeric so it's continuous" |
| Look at the actual histogram. What does it look like in your own words — lopsided, one hump or several, trailing off to one side? | The technical labels (symmetric, skewed, bounded, multimodal) are names for what you can already see; describing it first confirms you looked rather than fetched a term | the shape in the student's own words, per named feature |
| Is it long-tailed? | Long tails break means, standard deviations, and every standard outlier rule | yes/no, and what you will report instead |
| Are the extreme values errors, or ordinary members of a tail? | Removing tail members because they are large is the most common way to destroy a finding | the criterion separating the two |
| Is it multimodal — and if so, which hidden variable are you looking at? | Two peaks usually means two populations mixed | the candidate grouping variable |
| How many **independent units** sit behind each group you intend to compare? | Three people produce no summary stable enough to compare, however many rows they generated between them. And every set of numbers has a maximum, so one group is always "highest" | the count per group, at the claim's unit — not the row count |

**The gate matters more than the checklist.** Fitting a distribution to every column because
the data has that many columns produces ten plots and no decision any of them changed. Name
what the shape is for before looking at it; if nothing downstream would change either way,
skip it and say so.

**Discrete versus continuous is a choice, not a fact.** Almost any feature can be treated
either way — age as an integer or as "young/old", message length as a number or as
"short/medium/long". Asking "discrete or continuous?" as if the column dictates the answer
skips the actual decision, which is which framing serves the claim.

**Where it does matter, do it before the interesting part, not after.** A correlation between
two long-tailed variables is usually a correlation between their two largest observations,
and you will not see that from the scatter plot alone.

**Pseudoreplication** is the name for getting the last row wrong, and it is worth knowing,
because a named error is one you can catch in your own draft. It looks like this: a bar chart
of message length by gender, thousands of rows behind each bar, an overwhelming difference —
built from three men and two women. The bar chart has n=4000. The claim has n=5. Aggregate to
one number per person and plot those five points instead; the groups will usually overlap
completely. Same data, two units of analysis, opposite conclusions.

See [Distributions](05-distributions.md).

---

## 9.4 Stage 4 — The encoding

**Settle:** which family (or families) the question lives in, and which visual channel
carries the comparison you actually care about.

| Ask yourself | Why it matters | A good answer names |
|---|---|---|
| Which family is the obvious fit — and which other family could frame the same question, including one that feels like a stretch? | One phenomenon usually fits more than one family. The obvious framing is a good default, not a foregone conclusion, and the stretch is sometimes where the actual finding is | the obvious family and at least one real alternative, each with a reason it could show something the other hides |
| Sketch what each candidate would look like — which one actually shows the arc from stage 1 most clearly? | Cheap to sketch, expensive to build. Deciding after a rough version beats deciding from a mental picture of a chart nobody made yet | which sketch won, and what it showed that the runner-up didn't |
| Which comparison is the point? | The point gets position or length; everything else gets a weaker channel | the one comparison |
| What is on each axis, and what does the spacing mean? | Categorical axes are evenly spaced regardless of real distance | both axes, with units |
| What aggregation happened, and could it hide the finding? | A mean over a bimodal group describes nobody | the aggregation, and what it discards |
| Which parameter, if changed, would change the conclusion? | Bin count, smoothing window, perplexity, top-k cut — each is part of the claim | the parameter, and the range you checked |

**One family is a default, not a destination.** categories / time / distributions /
relations / high-dimensional. Choosing a chart type before the question is choosing an
answer first — but the question rarely pins down only one family, and a five-minute sketch
of the unlikely candidate sometimes beats the obvious one outright.

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
