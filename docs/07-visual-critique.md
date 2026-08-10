# 7. Visual critique

A plot is an argument. This chapter is how to judge whether it makes one.

Work through the four sections in order on any finished visualisation. The
`goad_critique_visual` MCP tool returns these same questions, one section at a time, so an
assistant can put them to you rather than answer them for you.

---

## 7.1 Preattentive processing — the first 200 milliseconds

Before a reader thinks, their visual system has already extracted certain features in
parallel, across the whole image, at no cognitive cost: **colour**, **size**, **position**,
**orientation**, **enclosure**, **motion**. Everything else — reading a label, comparing two
numbers, following a legend to a line — is serial and slow.

The design consequence: whatever your message is, it must be carried by a preattentive
channel. If understanding your plot requires the reader to match colours to a legend and then
compare, you have made them work for something you could have given them free.

**Questions to ask:**

- Look away, look back, and note what you saw first. Is that the message?
- What is the highest-contrast thing in the image? Is it the data, or is it a gridline, a
  border, a legend box, or the most saturated colour in a categorical palette?
- Is there anything bright, large or enclosed that is *not* important?
- Can the main comparison be made by position or length alone, without colour?

The most common failure is a rainbow palette across categories nobody is being asked to
compare. Every colour is shouting, so nothing is.

---

## 7.2 Gestalt principles — what the eye groups by itself

Readers see groups before they see elements, and they do it by rules you do not control. You
can either work with them or fight them.

| Principle | The rule | What it does for you |
|---|---|---|
| **Proximity** | Things near each other belong together | spacing between bar groups does more than any label |
| **Similarity** | Things that look alike belong together | same colour = same series, across every panel |
| **Enclosure** | Things inside a boundary belong together | a shaded region marks a period without adding a line |
| **Closure** | The eye completes incomplete shapes | you can drop chart junk; the frame is implied |
| **Continuity** | The eye follows a line and expects it to continue | which is why a gap plotted as a straight line lies so effectively |
| **Connection** | Connected things belong together, more strongly than colour or proximity | a line joining points asserts they are a sequence |

**Questions to ask:**

- What is grouped visually, and does it match what is grouped conceptually?
- Is anything accidentally grouped — two unrelated series in similar blues, two categories
  adjacent for alphabetical reasons?
- Are you *connecting* points that are not a sequence? A line implies continuity; between
  unordered categories, it asserts something false.
- Could enclosure or proximity replace a legend or a set of labels?

---

## 7.3 The five guidelines

**1. Show the data.** The data is the reason the plot exists. If a gridline, a border or a
background is competing with it, remove them. Aggregating away variation you have — a bar of
means over a distribution you could have shown — is also a failure to show the data.

**2. Reduce clutter.** Everything on the plot should be earning its place. Redundant axis
labels, decimal places nobody needs, a legend for two series that could be labelled directly,
a title restating the axis labels. Delete and check whether anything was lost.

**3. Integrate text.** Labels on the thing they refer to, not in a legend the reader has to
traverse. And the strongest form: put the *finding* on the plot as text. A vertical line at
the date something changed, annotated with what changed, is worth more than a paragraph
underneath. `VerticalDate` exists for exactly this.

**4. Avoid spaghetti.** Past about five overlapping lines, nobody is tracing any of them.
Options: small multiples (one panel per series), highlight one series and grey the rest, or
aggregate. Small multiples work far more often than people expect — `create_figure(n_plots=n)`
makes them cheap.

**5. Start with grey.** Draw everything in grey first. Then add colour to the one thing you
are talking about. Colour used this way is a pointer; colour used everywhere is decoration
and costs the reader attention with no return.

**Questions to ask:**

- Which single element carries the message? Is it the only coloured thing?
- What could you delete without losing meaning? Delete it.
- Is any text more than one saccade away from what it describes?
- If there are more than five lines: why?

---

## 7.4 What are you claiming?

The three sections above judge the picture. This one judges whether it says anything.

**Questions to ask:**

- **State the claim in one sentence.** If you cannot, the plot is exploration and does not
  belong in a report yet — which is fine, but know which one you are holding.
- **What would falsify it?** Point at the region of the plot where the opposite result would
  have appeared. If there is no such region, the plot cannot support the claim.
- **Does the plot show the comparison, or does the reader have to do it?** "Group A is higher
  than group B" should be visible as a difference, not computed from two numbers in different
  places.
- **What is deliberately not shown, and would a reader want it?** The confidence, the sample
  size behind each bar, the points behind the mean, the axis starting somewhere other than
  zero.
- **Would somebody who disagrees with you be persuaded, or would they say "that depends on
  the smoothing window / the bin count / the perplexity"?** If the second, that parameter is
  part of the claim and belongs on the plot.

---

## 7.5 Using this with an assistant

An assistant is genuinely good at §7.2 and §7.3 — it will reliably spot a rainbow palette,
an unsorted category axis, a legend that could be direct labels. Ask it to.

It is not able to do §7.4, because it does not know what you are claiming or who is going to
read it. If it answers §7.4 anyway, it is inventing a claim to fit the picture, which is
precisely backwards.

So the productive division: you state the claim, it critiques the execution. The
`goad_critique_visual` tool is written to enforce that split — it returns the questions and
instructs the assistant to ask you §7.1 and §7.4 rather than answer them.

---

Next: [API reference](08-api-reference.md).
