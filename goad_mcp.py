#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastmcp>=2.14.2"]
# ///
"""MCP server that exposes the GOAD course as analysis-coaching tools.

The course is a set of markdown chapters in `docs/`, indexed by the reading-order table in
`docs/README.md`. Beyond serving those chapters for lookup and search, this server coaches
two activities the docs describe but a chat window cannot enforce on its own —

- **analysing** a dataset from a question rather than from the columns, worked through as a
  six-stage conversation (question -> data -> shape -> encoding -> critique -> verification)
  rather than handed over as one document to skim;
- **critiquing** a finished visualisation against the same checklist a grader will use
  (preattentive processing, gestalt principles, the five guidelines, and what is actually
  being claimed).

The analysis conversation (`goad_analysis_checklist`) is deliberately stateful *within one
server process* (i.e. for the current session): it returns one stage's checklist at a time,
and only advances once the assistant reports back what the student actually said. The point
is to make the assistant think *with* the student against this method, rather than hand back
six plots and a conclusion nobody can defend.

It reads `docs/` from disk when they sit next to this script; when launched standalone (e.g.
via `uv run` straight from a URL, with no course files nearby) it fetches the same content
from the public repo over HTTP instead. No build step, no embeddings, no database.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Paths and course model
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent
DOCS_DIR = REPO_ROOT / "docs"
DOCS_README = DOCS_DIR / "README.md"

# If the course files aren't next to this script, fetch them from the public repo over HTTP.
# This is what makes the zero-clone one-liner work:
#   uv run --no-project <RAW_BASE>/goad_mcp.py
#
# GOAD_REF selects which git ref (branch, tag, or commit) to fetch content from; it defaults
# to "main". Point the launch URL at the same ref and both the script and the docs it fetches
# stay pinned together — see MCP_SERVER.md.
GOAD_REF = os.environ.get("GOAD_REF", "main")
RAW_BASE = f"https://raw.githubusercontent.com/raoulg/goad_toolkit/{GOAD_REF}"
LOCAL = DOCS_README.is_file() and DOCS_DIR.is_dir()

# Cache for content fetched over HTTP (unused in local mode).
_remote_cache: dict[str, str] = {}


def _load_text(rel_path: str) -> str:
    """Load a course file's text, relative to the repo root.

    e.g. `_load_text("docs/README.md")` or `_load_text("docs/02-pipelines.md")`.

    Local files are read fresh on each call, so edits show up without a restart. Remote files
    are fetched once and cached.
    """
    if LOCAL:
        return (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    if rel_path not in _remote_cache:
        with urllib.request.urlopen(f"{RAW_BASE}/{rel_path}", timeout=30) as resp:  # noqa: S310
            _remote_cache[rel_path] = resp.read().decode("utf-8")
    return _remote_cache[rel_path]


mcp = FastMCP("goad_mcp")


# --------------------------------------------------------------------------- #
# Loading the chapter index from docs/README.md's reading-order table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Chapter:
    """One row of the docs/README.md reading-order table."""

    number: int  # 1..10
    title: str  # human-readable chapter title, e.g. "Pipelines"
    file: str  # doc filename, e.g. "02-pipelines.md"
    summary: str  # one-line "what it covers"


# Matches a reading-order table row:
#   | 2 | [Pipelines](02-pipelines.md) | TransformBase, Pipeline, ... |
_ROW = re.compile(
    r"^\|\s*(?P<num>\d+)\s*\|\s*\[(?P<title>.+?)\]\((?P<file>.+?\.md)\)\s*\|\s*(?P<summary>.+?)\s*\|\s*$"
)


def _load_chapters() -> list[Chapter]:
    """Parse docs/README.md's reading-order table into an ordered list of Chapters."""
    chapters: list[Chapter] = []
    for line in _load_text("docs/README.md").splitlines():
        row = _ROW.match(line)
        if not row:
            continue
        chapters.append(
            Chapter(
                number=int(row.group("num")),
                title=row.group("title"),
                file=row.group("file"),
                summary=row.group("summary"),
            )
        )
    return chapters


# Loaded once at startup; the corpus is tiny and static per process.
CHAPTERS: list[Chapter] = _load_chapters()
# Map a few forms of a name/filename/number to its Chapter for forgiving lookups.
_BY_KEY: dict[str, Chapter] = {}
for _c in CHAPTERS:
    _BY_KEY[str(_c.number)] = _c
    _BY_KEY[f"{_c.number:02d}"] = _c
    _BY_KEY[_c.file.lower()] = _c
    _BY_KEY[_c.file[:-3].lower()] = _c  # without ".md"
    _BY_KEY[_c.title.lower()] = _c


def _find_chapter(query: str) -> Optional[Chapter]:
    """Resolve a chapter by number, filename, stem, or title (case-insensitive)."""
    key = query.strip().lower()
    if key in _BY_KEY:
        return _BY_KEY[key]
    # Fall back to a partial match on the title.
    for chapter in CHAPTERS:
        if key in chapter.title.lower():
            return chapter
    return None


def _must_find_chapter(query: str) -> Chapter:
    """Like _find_chapter, for internal callers that name a chapter known to exist.

    Used only for the fixed chapter numbers this module cites in its own coaching text (e.g.
    "01", "09") — never for user-supplied input, which goes through _find_chapter and handles
    None explicitly.
    """
    chapter = _find_chapter(query)
    assert chapter is not None, f"expected chapter '{query}' to exist in docs/README.md"  # noqa: S101
    return chapter


def _doc_url(filename: str, anchor: str = "") -> str:
    """Browsable GitHub URL for a chapter (rendered markdown, for humans).

    Uses the same ref the server is pinned to, so a link always points at the exact text the
    reader is being coached on.
    """
    suffix = f"#{anchor}" if anchor else ""
    return f"https://github.com/raoulg/goad_toolkit/blob/{GOAD_REF}/docs/{filename}{suffix}"


def _chapter_ref(chapter: Chapter) -> str:
    """A markdown link to a chapter's online text."""
    return f"[{chapter.file}]({_doc_url(chapter.file)})"


def _read_doc(filename: str) -> str:
    """Read a chapter's text, raising a clear error if it is missing.

    Works whether the docs are local or fetched from the public repo; any failure to obtain
    the file is normalised to FileNotFoundError.
    """
    try:
        return _load_text(f"docs/{filename}")
    except (OSError, urllib.error.URLError) as exc:
        raise FileNotFoundError(filename) from exc


# --------------------------------------------------------------------------- #
# Input models — chapter lookup
# --------------------------------------------------------------------------- #


class ListConceptsInput(BaseModel):
    """Input for listing course chapters."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    keyword: Optional[str] = Field(
        default=None,
        description=(
            "Optional substring filter, matched case-insensitively against each chapter's "
            "title and one-line summary. Omit to list all chapters."
        ),
        max_length=200,
    )


class GetConceptInput(BaseModel):
    """Input for fetching a single chapter."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chapter: str = Field(
        ...,
        description=(
            "Chapter to fetch. Accepts the number ('4' or '04'), the doc filename "
            "('04-five-families.md'), its stem ('04-five-families'), or the title "
            "('The five families')."
        ),
        min_length=1,
        max_length=200,
    )


class SearchInput(BaseModel):
    """Input for searching across all chapters."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Keyword(s) to search for across every chapter's text.",
        min_length=2,
        max_length=200,
    )
    limit: int = Field(
        default=5,
        description="Maximum number of matching chapters to return.",
        ge=1,
        le=20,
    )


# --------------------------------------------------------------------------- #
# Tools — chapter lookup
# --------------------------------------------------------------------------- #


@mcp.tool(
    name="goad_list_concepts",
    annotations={
        "title": "List GOAD course chapters",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def goad_list_concepts(params: ListConceptsInput) -> str:
    """List the GOAD course chapters, in reading order.

    Use this to discover what the course covers before fetching a chapter with
    goad_get_concept, or before calling goad_analysis_checklist / goad_critique_visual /
    goad_distribution_method to see which chapters they draw on.

    Args:
        params (ListConceptsInput):
            - keyword (Optional[str]): if set, keep only chapters whose title or summary
              contains this substring (case-insensitive).

    Returns:
        str: Markdown numbered list. Each line has the chapter number, title, filename link,
        and one-line summary.
    """
    chapters = CHAPTERS
    header = "# GOAD course chapters"
    if params.keyword:
        key = params.keyword.lower()
        chapters = [
            c for c in chapters if key in c.title.lower() or key in c.summary.lower()
        ]
        header = f"# GOAD chapters matching '{params.keyword}'"

    lines = [header, ""]
    for c in chapters:
        lines.append(f"{c.number}. **{c.title}** — {_chapter_ref(c)} — {c.summary}")
    if not chapters:
        lines.append("(no chapters matched)")
    return "\n".join(lines)


@mcp.tool(
    name="goad_get_concept",
    annotations={
        "title": "Get a GOAD course chapter",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def goad_get_concept(params: GetConceptInput) -> str:
    """Fetch the full markdown of one GOAD course chapter.

    Args:
        params (GetConceptInput):
            - chapter (str): number, filename, stem, or title of the chapter.

    Returns:
        str: The chapter's full markdown, prefixed with a link back to it online. If the
        chapter is not found, returns an error listing valid chapter names.
    """
    chapter = _find_chapter(params.chapter)
    if chapter is None:
        names = ", ".join(f"{c.number}:{c.title}" for c in CHAPTERS)
        return (
            f"Error: no chapter matching '{params.chapter}'. "
            f"Call goad_list_concepts first. Known chapters: {names}"
        )
    try:
        body = _read_doc(chapter.file)
    except FileNotFoundError:
        return f"Error: doc file '{chapter.file}' is listed in docs/README.md but missing on disk."

    header = f"_Further reading (share this link): {_doc_url(chapter.file)}_\n\n"
    return header + body


@mcp.tool(
    name="goad_search",
    annotations={
        "title": "Search GOAD course chapters",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def goad_search(params: SearchInput) -> str:
    """Search every chapter for keyword(s) and return the best matches.

    Ranking is simple term-frequency: chapters are scored by how many times the query terms
    appear. Good for "where do the docs talk about X?" — e.g. "long tail", "plot_on",
    "Simpson", "perplexity", "residual".

    Args:
        params (SearchInput):
            - query (str): one or more keywords.
            - limit (int): max chapters to return (1-20, default 5).

    Returns:
        str: Markdown list of matching chapters, each with a match count and a short snippet
        around the first hit. "No matches" if nothing is found.
    """
    terms = [t for t in params.query.lower().split() if t]
    scored: list[tuple[int, Chapter, str]] = []

    for chapter in CHAPTERS:
        try:
            text = _read_doc(chapter.file)
        except FileNotFoundError:
            continue
        lower = text.lower()
        score = sum(lower.count(term) for term in terms)
        if score == 0:
            continue
        first = min(
            (lower.find(term) for term in terms if lower.find(term) != -1),
            default=-1,
        )
        start = max(0, first - 80)
        snippet = text[start : first + 120].replace("\n", " ").strip()
        scored.append((score, chapter, snippet))

    if not scored:
        return f"No matches for '{params.query}'."

    scored.sort(key=lambda item: item[0], reverse=True)
    lines = [f"# Search results for '{params.query}'", ""]
    for score, chapter, snippet in scored[: params.limit]:
        lines.append(f"## {chapter.title} — {score} hit(s)")
        lines.append(f"Further reading: {_doc_url(chapter.file)}")
        lines.append(f"…{snippet}…")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The analysis conversation — six stages, walked through one at a time
# --------------------------------------------------------------------------- #


class AnalysisStage(str, Enum):
    """A stage in the analysis conversation, in the order it must happen."""

    question = "question"
    data = "data"
    shape = "shape"
    encoding = "encoding"
    critique = "critique"
    verification = "verification"


STAGE_ORDER: list[AnalysisStage] = [
    AnalysisStage.question,
    AnalysisStage.data,
    AnalysisStage.shape,
    AnalysisStage.encoding,
    AnalysisStage.critique,
    AnalysisStage.verification,
]

STAGE_TITLE: dict[AnalysisStage, str] = {
    AnalysisStage.question: "1. What do you actually want to know? (question)",
    AnalysisStage.data: "2. What is one row, and what is missing? (data)",
    AnalysisStage.shape: "3. What shape is each variable in? (shape)",
    AnalysisStage.encoding: "4. Which family, which channel? (encoding)",
    AnalysisStage.critique: "5. Does the picture make the argument? (critique)",
    AnalysisStage.verification: "6. How would you know it is real? (verification)",
}


def _goad() -> str:
    return _chapter_ref(_must_find_chapter("01"))


def _pipelines() -> str:
    return _chapter_ref(_must_find_chapter("02"))


def _families() -> str:
    return _chapter_ref(_must_find_chapter("04"))


def _dists() -> str:
    return _chapter_ref(_must_find_chapter("05"))


def _models() -> str:
    return _chapter_ref(_must_find_chapter("06"))


def _critique_ch() -> str:
    return _chapter_ref(_must_find_chapter("07"))


def _method() -> str:
    return _chapter_ref(_must_find_chapter("09"))


STAGE_INTRO: dict[AnalysisStage, str] = {
    AnalysisStage.question: (
        "Nothing downstream can be decided until this is settled. An analysis that starts "
        "from the columns produces a tour of the columns; an analysis that starts from a "
        f"question produces an answer somebody can disagree with ({_goad()} §1.2)."
    ),
    AnalysisStage.data: (
        "Now — and only now — look at what you have. The gap between what the data contains "
        "and what the question needs is the enrichment step, and it is usually where the "
        f"analysis is won rather than in any later method ({_pipelines()})."
    ),
    AnalysisStage.shape: (
        "Before comparing, correlating or modelling anything, find out how each variable is "
        "distributed. A correlation between two long-tailed variables is often a correlation "
        f"between their two largest observations ({_dists()})."
    ),
    AnalysisStage.encoding: (
        "The family follows from the question, not from the chart you already pictured. "
        f"Choosing a chart type first is choosing an answer first ({_families()})."
    ),
    AnalysisStage.critique: (
        "Judge the picture the way a reader will: what it says in the first 200ms, what the "
        f"eye groups by itself, and what is actually being claimed ({_critique_ch()})."
    ),
    AnalysisStage.verification: (
        "A pattern you found is not yet a pattern you can report. This stage is what turns "
        f"the difference into a claim ({_method()} §9.6)."
    ),
}

# Each row: (question, why it matters, what a good answer names)
StageRow = tuple[str, str, str]

STAGE_ROWS: dict[AnalysisStage, list[StageRow]] = {
    AnalysisStage.question: [
        (
            "What do you want to know, in one sentence?",
            "Every later decision — which transform, which plot, which test — is downstream "
            "of this, and it is the stage students skip.",
            "a proposition, not a topic ('are messages concentrated in two daily peaks', "
            "not 'message timing')",
        ),
        (
            "What result would make you say 'no'?",
            "A question no result can contradict cannot be answered, only decorated.",
            "the specific pattern whose absence would settle it",
        ),
        (
            "Who reads the answer, and how long do they have?",
            "Decides how much has to be visible in the picture versus how much can stay in "
            "your head.",
            "an audience and a rough time budget",
        ),
        (
            "Does anyone care what the answer turns out to be?",
            "If both outcomes are equally uninteresting, better to find that out now than "
            "after a week of work.",
            "a consequence that differs depending on the outcome",
        ),
    ],
    AnalysisStage.data: [
        (
            "What is one row?",
            "If you cannot say it in a sentence, later aggregations will be wrong in ways "
            "that do not raise an error.",
            "the unit of observation, explicitly",
        ),
        (
            "Which columns were measured, and which were derived — by whom, and how?",
            "A derived column carries somebody's assumptions, including yours from an hour "
            "ago.",
            "the provenance of each column the question depends on",
        ),
        (
            "What does your question need that the data does not yet contain?",
            "This is the enrichment step. A hand-written regex or timestamp feature "
            "regularly beats a more sophisticated method on the same raw columns.",
            "the specific derived features, and how each would be computed",
        ),
        (
            "What is missing, and is the missingness itself informative?",
            "A silent gap is a data point about the process, not an inconvenience.",
            "the pattern of missingness, not just the count",
        ),
        (
            f"Which of these steps belong in a Pipeline rather than a notebook cell? "
            f"({_pipelines()} §2.7)",
            "Anything you would want to run again on new data. Exploration stays a cell; "
            "enrichment becomes a named step.",
            "the steps, named",
        ),
    ],
    AnalysisStage.shape: [
        (
            "For each variable in play: discrete or continuous?",
            "Selects which half of the distribution registry can even apply, and the fitter "
            "will not tell you if you get it wrong.",
            "which, per variable",
        ),
        (
            "What does each histogram look like — symmetric, skewed, bounded, multimodal?",
            "A mean and a standard deviation summarise exactly one of those honestly.",
            "the shape, in words, per variable",
        ),
        (
            "Is anything long-tailed?",
            "Long tails break means, standard deviations, and every standard outlier rule "
            "(z>3, 1.5*IQR) — all of which quietly assume normality.",
            "yes/no, and what you will report instead of a mean",
        ),
        (
            f"Are the extreme values errors, or ordinary members of a tail? ({_dists()} §5.6)",
            "Removing tail members because they are large is the most common way to delete "
            "the finding.",
            "the criterion that separates 'could not have been produced' from 'unusual'",
        ),
        (
            "Is anything multimodal — and if so, which hidden variable are you looking at?",
            "Two peaks usually means two populations mixed together.",
            "the candidate grouping variable",
        ),
    ],
    AnalysisStage.encoding: [
        (
            f"Which of the five families does this question live in? ({_families()})",
            "categories / time / distributions / relations / high-dimensional. The family "
            "follows from the question.",
            "one family, and why the question belongs to it",
        ),
        (
            "Which single comparison is the point of the plot?",
            "The point gets position or length — the channels the eye reads for free. "
            "Everything else gets a weaker channel or gets cut.",
            "the one comparison, stated",
        ),
        (
            "What is on each axis, and what does the spacing mean?",
            "A categorical axis is evenly spaced regardless of the real distance between "
            "its values; a datetime axis is not.",
            "both axes, with units, and which kind of axis each is",
        ),
        (
            "What aggregation happened on the way to this plot, and could it hide the thing "
            "you are looking for?",
            "A mean over a bimodal group describes nobody in it.",
            "the aggregation, and what it discards",
        ),
        (
            "Which parameter, if you changed it, would change the conclusion?",
            "Bin count, smoothing window, top-k cut, perplexity — each of these is part of "
            "the claim whether or not you mention it.",
            "the parameter, and the range you intend to check",
        ),
    ],
    AnalysisStage.critique: [
        (
            "Look away from the plot, look back, and say what you saw first.",
            "The preattentive channels (colour, size, position, enclosure) carry your "
            "message whether or not you chose them deliberately.",
            "what caught the eye, and whether that is the message",
        ),
        (
            f"What is grouped visually, and does it match what is grouped conceptually? "
            f"({_critique_ch()} §7.2)",
            "Proximity, similarity, enclosure and connection group things for the reader "
            "without asking your permission.",
            "the groupings — both the intended ones and the accidental ones",
        ),
        (
            "Which single element carries the message, and is it the only coloured thing?",
            "Grey first, then colour as a pointer. Colour everywhere is decoration, and it "
            "costs the reader attention with no return.",
            "the one element",
        ),
        (
            "What could you delete without losing meaning?",
            "Gridlines, borders, legends replaceable by direct labels, decimal places "
            "nobody needs. Everything remaining should be earning its place.",
            "the list of things deleted",
        ),
        (
            "State the claim this plot supports, in one sentence. Then: what would falsify "
            "it, and where on the plot would that have shown up?",
            "If there is no region of the plot where the opposite result would have "
            "appeared, the plot cannot support the claim.",
            "the claim, and the region that would have contradicted it",
        ),
    ],
    AnalysisStage.verification: [
        (
            "What model of 'nothing interesting is happening' would also produce this?",
            "That null is what your claim is measured against. Without it, any pattern is "
            "as good as any other.",
            "the null, stated concretely",
        ),
        (
            "If you shuffled the labels and re-measured, how often would you see something "
            "this strong?",
            "The cheapest honest check available, and it needs no distributional theory at "
            "all.",
            "what gets shuffled, and a rough frequency",
        ),
        (
            "Is there a third variable that could produce this pattern on its own?",
            "The most common route from a real correlation to a false mechanism. Two "
            "variables that both grow will correlate; that is arithmetic.",
            "the candidate confounder, and how you checked it",
        ),
        (
            "Does the pattern survive on a slice you did not use to find it?",
            "A pattern found and confirmed on the same data has not been confirmed.",
            "the held-out slice, and the result on it",
        ),
        (
            f"If you modelled it: what does the residual look like over time, and what "
            f"distribution fits it? ({_models()} §6.3)",
            "Structure left in the residual is an unmodelled mechanism, not noise — and "
            "*when* it starts is usually more informative than any summary of it.",
            "the residual's shape over time, and its fitted family",
        ),
    ],
}

# In-memory record of what the student said at each stage, for THIS server process (i.e.
# this session). Resets when the server restarts — an analysis conversation is meant to
# happen in one sitting, not to resume silently across days.
_progress: dict[str, str] = {}

MIN_ANSWER_CHARS = 20


def _next_stage(stage: AnalysisStage) -> Optional[AnalysisStage]:
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None


def _roadmap() -> str:
    lines = [
        "# Analysing data with GOAD — a six-stage conversation",
        "",
        "This is deliberately not a single document to skim. Work through these six stages "
        "with the student, in order, one call to this tool per stage — each later stage "
        "assumes the earlier ones were actually reasoned through together, not skipped.",
        "",
    ]
    for stage in STAGE_ORDER:
        mark = "✅" if stage.value in _progress else "▢"
        lines.append(f"{mark} **{STAGE_TITLE[stage]}** (`{stage.value}`)")
    lines += [
        "",
        'Call this tool again with `stage="question"` to begin — always start there, even '
        "when the student arrives with a dataset and an idea of the plot they want. Stage 4 "
        "depends on what stage 1 actually established.",
    ]
    return "\n".join(lines)


def _stage_table(stage: AnalysisStage) -> str:
    rows = STAGE_ROWS[stage]
    lines = [
        f"# Stage: {STAGE_TITLE[stage]}",
        "",
        STAGE_INTRO[stage],
        "",
        "| Question | Why it matters | A good answer names |",
        "|---|---|---|",
    ]
    for q, why, names in rows:
        lines.append(f"| {q} | {why} | {names} |")
    lines += [
        "",
        "**How to use this table:** ask the student these questions yourself — together or "
        "one at a time, your judgement. Do NOT answer them yourself, and do not treat this "
        "stage as done until they have engaged in their own words. Once they have, call "
        f'`goad_analysis_checklist` again with `stage="{stage.value}"` and '
        "`answers=<a summary of what they actually said>` to record it and receive the next "
        "stage.",
    ]
    return "\n".join(lines)


def _completion_summary() -> str:
    lines = [
        "# All six stages recorded",
        "",
        "What was reasoned through, in order — this is a usable first draft of the "
        "student's write-up:",
        "",
    ]
    for stage in STAGE_ORDER:
        lines.append(f"**{STAGE_TITLE[stage]}**")
        lines.append(f"> {_progress.get(stage.value, '(not recorded)')}")
        lines.append("")
    lines += [
        "Now is the right time to write code, if the student wants it — and it will be code "
        "with a stated purpose rather than a tour of the columns.",
        "",
        "If the verification stage left the pattern in doubt, the honest next step is a "
        "sharper version of stage 1 rather than a better plot. Findings come from the second "
        f"lap ({_method()} §9.7).",
    ]
    return "\n".join(lines)


class AnalysisChecklistInput(BaseModel):
    """Input for the staged analysis-checklist tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    stage: Optional[AnalysisStage] = Field(
        default=None,
        description=(
            "Which stage to open. Omit on the very first call to get the roadmap and confirm "
            "where to start (always 'question'). One of: question, data, shape, encoding, "
            "critique, verification."
        ),
    )
    answers: Optional[str] = Field(
        default=None,
        description=(
            "A summary, in your own words, of what the student actually said when asked this "
            "stage's questions. Omit to just receive the stage's checklist. Provide it on a "
            "second call for the SAME stage to record it and advance to the next stage. Do "
            "not fabricate this on the student's behalf — it must reflect their real answers."
        ),
        max_length=4000,
    )


@mcp.tool(
    name="goad_analysis_checklist",
    annotations={
        "title": "Walk through the GOAD analysis method with a student, one stage at a time",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def goad_analysis_checklist(params: AnalysisChecklistInput) -> str:
    """Coach a student through six stages of analysing data, from a question to a claim.

    This is a conversation, not a document dump: each call returns exactly ONE stage's
    checklist table. Ask the student those questions yourself, in the chat, and wait for real
    answers — do not answer on their behalf. Once they have responded, call this tool again
    with the SAME `stage` and their answers summarised in `answers`; that records the stage
    and returns the next one. Progress is held in memory for this server process only (this
    session) and is not persisted across restarts.

    The six stages, in the order they must happen:
        1. question     — what do you want to know, and what would count as an answer?
        2. data         — what is one row, and what does the question need that is missing?
        3. shape        — how is each variable distributed, before you compare anything?
        4. encoding     — which of the five families, and which visual channel carries it?
        5. critique     — does the picture make the argument a reader will read?
        6. verification — how would you know the pattern is real?

    Args:
        params (AnalysisChecklistInput):
            - stage (Optional[AnalysisStage]): which stage to open; omit for the roadmap.
            - answers (Optional[str]): the student's answers to the CURRENT stage, in your
              words; omit to just receive the stage's checklist.

    Returns:
        str: The roadmap (no stage given), a stage's checklist table (stage given, no answers
        yet), or a recorded confirmation plus the next stage's checklist (stage + answers
        given). After the last stage, returns a recap of everything recorded.
    """
    if params.stage is None:
        return _roadmap()

    stage = params.stage
    if not params.answers or len(params.answers.strip()) < MIN_ANSWER_CHARS:
        return _stage_table(stage)

    _progress[stage.value] = params.answers.strip()
    nxt = _next_stage(stage)
    if nxt is None:
        return _completion_summary()

    recap = f"✅ Recorded for **{STAGE_TITLE[stage]}**: {params.answers.strip()}\n\n"
    return recap + _stage_table(nxt)


# --------------------------------------------------------------------------- #
# Visual critique — the checklist a grader will use, one section at a time
# --------------------------------------------------------------------------- #


class CritiqueSection(str, Enum):
    """A section of the visual-critique checklist, in the order to apply it."""

    first_impression = "first_impression"
    grouping = "grouping"
    guidelines = "guidelines"
    claim = "claim"


CRITIQUE_ORDER: list[CritiqueSection] = [
    CritiqueSection.first_impression,
    CritiqueSection.grouping,
    CritiqueSection.guidelines,
    CritiqueSection.claim,
]

CRITIQUE_TITLE: dict[CritiqueSection, str] = {
    CritiqueSection.first_impression: "1. The first 200 milliseconds (preattentive processing)",
    CritiqueSection.grouping: "2. What the eye groups by itself (gestalt principles)",
    CritiqueSection.guidelines: "3. The five guidelines",
    CritiqueSection.claim: "4. What are you claiming?",
}

CRITIQUE_INTRO: dict[CritiqueSection, str] = {
    CritiqueSection.first_impression: (
        "Before a reader thinks, their visual system has already extracted colour, size, "
        "position, orientation and enclosure across the whole image, for free. Everything "
        "else — reading a label, following a legend to a line — is slow. Whatever the "
        "message is, it has to ride on one of the free channels."
    ),
    CritiqueSection.grouping: (
        "Readers see groups before they see elements, by rules you do not control. You can "
        "work with them or fight them."
    ),
    CritiqueSection.guidelines: (
        "Show the data / reduce clutter / integrate text / avoid spaghetti / start with grey."
    ),
    CritiqueSection.claim: (
        "The first three sections judge the picture. This one judges whether it says "
        "anything. It is also the section an assistant cannot answer for the student — it "
        "does not know what is being claimed or who is reading."
    ),
}

CRITIQUE_ROWS: dict[CritiqueSection, list[StageRow]] = {
    CritiqueSection.first_impression: [
        (
            "Look away, look back. What did you see first — and is that the message?",
            "This is the only test of preattentive design that works: your own eye, cold.",
            "what caught the eye, and whether it should have",
        ),
        (
            "What is the highest-contrast thing in the image?",
            "If it is a gridline, a border, a legend box or the most saturated colour in a "
            "categorical palette, the data is losing to its own decoration.",
            "the element, and whether it is data",
        ),
        (
            "Is anything bright, large or enclosed that is NOT important?",
            "Every one of those is a free channel spent on nothing.",
            "the list, or an explicit 'no'",
        ),
        (
            "Can the main comparison be made by position or length alone, without colour?",
            "Position and length are the most accurately read channels; colour is the least.",
            "yes, or which channel is carrying it instead",
        ),
    ],
    CritiqueSection.grouping: [
        (
            "What is grouped visually, and does it match what is grouped conceptually?",
            "Proximity and similarity group things for the reader whether or not you meant "
            "them to.",
            "the visual groupings, and the conceptual ones",
        ),
        (
            "Is anything accidentally grouped?",
            "Two unrelated series in similar blues; two categories adjacent because the "
            "sort was alphabetical.",
            "the accidental grouping, or an explicit 'none'",
        ),
        (
            "Are you connecting points that are not a sequence?",
            "A line asserts continuity. Between unordered categories, that assertion is "
            "false — and it is the most persuasive lie a chart can tell.",
            "yes/no, and what the connection implies",
        ),
        (
            "Could enclosure or proximity replace a legend or a set of labels?",
            "A shaded region or a direct label removes a lookup the reader would otherwise "
            "have to perform.",
            "the substitution, if there is one",
        ),
    ],
    CritiqueSection.guidelines: [
        (
            "SHOW THE DATA: is anything competing with it, and did you aggregate away "
            "variation you actually have?",
            "A bar of means over a distribution you could have shown is a failure to show "
            "the data.",
            "what competes, and what the aggregation hid",
        ),
        (
            "REDUCE CLUTTER: what can you delete without losing meaning?",
            "Delete it, then check whether anything was lost. Usually nothing was.",
            "the list deleted",
        ),
        (
            "INTEGRATE TEXT: is any label more than one glance away from what it describes "
            "— and is the finding itself written on the plot?",
            "A reader with thirty seconds reads the annotation, not the axes. A labelled "
            "vertical line at the moment something changed beats a paragraph underneath.",
            "where the text sits, and what it says",
        ),
        (
            "AVOID SPAGHETTI: if there are more than about five lines, why?",
            "Past five overlapping lines nobody traces any of them. Small multiples, "
            "highlight-one-and-grey-the-rest, or aggregate.",
            "the count, and the chosen alternative",
        ),
        (
            "START WITH GREY: is colour a pointer to the one thing you are talking about, "
            "or is it decoration?",
            "Colour used everywhere costs the reader attention and returns nothing.",
            "which element is coloured, and why that one",
        ),
    ],
    CritiqueSection.claim: [
        (
            "State the claim in one sentence.",
            "If you cannot, this plot is exploration — which is fine, but it is not a "
            "result yet.",
            "one sentence, with its scope",
        ),
        (
            "What would falsify it, and where on the plot would that have appeared?",
            "If there is no such region, the plot cannot support the claim.",
            "the falsifying pattern, and its location on the plot",
        ),
        (
            "Does the plot SHOW the comparison, or does the reader have to compute it?",
            "'A is higher than B' should be visible as a difference, not derived from two "
            "numbers in different places.",
            "how the comparison is shown",
        ),
        (
            "What is deliberately not shown, and would a reader want it?",
            "Sample size behind each bar, the points behind the mean, an axis that does not "
            "start at zero.",
            "what was omitted, and why that is defensible",
        ),
        (
            "Would somebody who disagrees be persuaded — or would they say 'that depends on "
            "the smoothing window / bin count / perplexity'?",
            "If the second, that parameter is part of the claim and belongs on the plot.",
            "the parameter the claim rests on, and the sweep that supports it",
        ),
    ],
}


class CritiqueInput(BaseModel):
    """Input for the visual-critique tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    section: Optional[CritiqueSection] = Field(
        default=None,
        description=(
            "Which section of the checklist to open. Omit on the first call to get the "
            "roadmap. One of: first_impression, grouping, guidelines, claim."
        ),
    )


@mcp.tool(
    name="goad_critique_visual",
    annotations={
        "title": "Critique a visualisation against the course checklist",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def goad_critique_visual(params: CritiqueInput) -> str:
    """Return the visual-critique checklist, one section at a time.

    This is the same checklist a grader uses. Work through the four sections with the student
    in order, one call per section.

    The division of labour matters here. Sections 2 and 3 (gestalt, the five guidelines) are
    ones you can genuinely help with — spot the rainbow palette, the unsorted category axis,
    the legend that could be direct labels. Sections 1 and 4 (what caught the eye, what is
    being claimed) must be answered by the STUDENT: you cannot see the plot the way they do,
    and you do not know what they are claiming or who is reading. Answering section 4 for
    them means inventing a claim to fit the picture, which is exactly backwards.

    Args:
        params (CritiqueInput):
            - section (Optional[CritiqueSection]): which section to open; omit for the
              roadmap.

    Returns:
        str: The roadmap (no section given), or one section's question table plus a pointer
        to the next section.
    """
    if params.section is None:
        lines = [
            "# Critiquing a visualisation — four sections",
            "",
            "Work through these with the student in order, one call per section. Ask the "
            "questions; do not answer sections 1 and 4 on their behalf.",
            "",
        ]
        for sec in CRITIQUE_ORDER:
            lines.append(f"- **{CRITIQUE_TITLE[sec]}** (`{sec.value}`)")
        lines += [
            "",
            f'Call this tool again with `section="first_impression"` to begin. Full chapter: '
            f"{_doc_url(_must_find_chapter('07').file)}",
        ]
        return "\n".join(lines)

    section = params.section
    lines = [
        f"# {CRITIQUE_TITLE[section]}",
        "",
        CRITIQUE_INTRO[section],
        "",
        "| Question | Why it matters | A good answer names |",
        "|---|---|---|",
    ]
    for q, why, names in CRITIQUE_ROWS[section]:
        lines.append(f"| {q} | {why} | {names} |")

    idx = CRITIQUE_ORDER.index(section)
    lines.append("")
    if section in (CritiqueSection.first_impression, CritiqueSection.claim):
        lines.append(
            "**These are the student's to answer.** You cannot see the plot the way they do, "
            "and you do not know what they are claiming. Ask, and wait."
        )
    else:
        lines.append(
            "**You can help here.** Point at specific elements — the palette, the sort "
            "order, the legend — and propose concrete fixes. Then let the student decide."
        )
    if idx + 1 < len(CRITIQUE_ORDER):
        nxt = CRITIQUE_ORDER[idx + 1]
        lines.append("")
        lines.append(
            f'When this section is worked through, call again with `section="{nxt.value}"`.'
        )
    else:
        lines.append("")
        lines.append(
            "That is the whole checklist. If section 4 could not be answered, the plot is "
            "exploration rather than a result — which is worth saying out loud before it "
            "ends up in a report."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The distribution method — an ordered checklist for fitting and defending a distribution
# --------------------------------------------------------------------------- #


class DistributionMethodInput(BaseModel):
    """Input for the distribution-method tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    variable: str = Field(
        default="my variable",
        description=(
            "One short description of the variable being fitted, e.g. 'seconds between "
            "messages'. Purely for framing the returned method's header; the method itself "
            "is generic."
        ),
        max_length=300,
    )


@mcp.tool(
    name="goad_distribution_method",
    annotations={
        "title": "Get the method for fitting and defending a distribution",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def goad_distribution_method(params: DistributionMethodInput) -> str:
    """Return the ordered method for going from a column of numbers to a defended claim.

    Call this when the student has a variable and wants to say something about how it is
    distributed — how typical a value is, whether something is an outlier, or what process
    might have generated it. This returns the *order of operations*; fetch the full chapter
    with goad_get_concept('05') when they are ready for the detail.

    Args:
        params (DistributionMethodInput):
            - variable (str): short description of what is being fitted, used to frame the
              header.

    Returns:
        str: Markdown checklist, in the order the steps must happen, each linked to the
        section of the distributions chapter it comes from.
    """
    ch = _must_find_chapter("05")
    url = _doc_url(ch.file)

    lines = [
        f"# Fitting a distribution to: {params.variable}",
        "",
        "Do these in order. Reversing them — in particular, deciding what an outlier is "
        f"before knowing what family the data follows — is how findings get deleted. Full "
        f"chapter: {url}",
        "",
        "1. **Look at it first, without summarising.** Histogram and ECDF. Symmetric or "
        "skewed? Bounded? One peak or several? Multiple peaks usually mean two populations "
        "mixed, which is a grouping question, not a fitting one.",
        "",
        "2. **Decide discrete or continuous, and commit.** Counts are discrete; durations "
        "are continuous. `DistributionFitter.fit(data, discrete=...)` will silently fit the "
        "wrong half of the registry if this is wrong.",
        "",
        "3. **Say what you expect, before fitting.** Waiting times suggest exponential; "
        "counts of rare events suggest Poisson; products of factors suggest lognormal; "
        "rank-frequency suggests a power law. A prediction you can be wrong about makes the "
        f"fit informative rather than decorative. ({url}#57-what-the-families-mean)",
        "",
        "4. **Check the registry has the family you need, and register it if not.** The "
        "shipped set has no pareto, no bernoulli, no negative binomial — so a power-law "
        "question fitted against the defaults will come back 'lognormal' and look "
        f"convincing. ({url}#52-the-registry)",
        "",
        "5. **Fit the whole registry, not one family.** The comparison is the evidence. "
        "Failures come back as `FailedFit` values rather than exceptions, so one badly "
        "behaved family does not cost you the rest.",
        "",
        "6. **Read KS and log-likelihood as two different questions.** Log-likelihood is "
        "about the bulk; KS is about the largest gap in the CDF and is comparatively blind "
        "in the tails. When they disagree, the shape is systematically off somewhere — go "
        f"find where. ({url}#54-ks-versus-log-likelihood)",
        "",
        "7. **Look at the picture, then the tails specifically.** `PlotFits` ranks the "
        "candidates and colours the two winners. Systematic deviation matters; bar-by-bar "
        "wobble does not. If the question is about extreme values, a histogram will not "
        f"settle it — use a qq-plot or an ECDF on log axes. ({url}#55-reading-plotfits)",
        "",
        "8. **Only now decide what an outlier is.** A point at the 99.9th percentile of a "
        "fitted exponential is expected in a sample of a few thousand; the same point is a "
        "five-sigma event under a normal fit. Separate 'could not have been produced by this "
        f"process' from 'unusual but possible'. ({url}#56-outliers-properly)",
        "",
        "9. **Report what you removed, if anything** — how many, on what criterion, and what "
        "the analysis looks like with them left in. An analysis that changes conclusion when "
        "three points are removed is a finding about those three points.",
        "",
        "10. **If this was a residual, read the structure, not just the shape.** A skewed "
        "residual means the model is systematically wrong in one direction; a residual with "
        "a step or a drift in it is an unmodelled mechanism, and locating *when* it starts "
        f"is worth more than any summary of it. ({url}#58-fitting-the-residual)",
        "",
        "Call `goad_get_concept('05')` for the full chapter, and "
        "`goad_get_concept('06')` for the model/residual loop that step 10 belongs to.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


@mcp.prompt(
    name="analyse",
    description="Start a six-stage analysis conversation, from a question to a defensible claim.",
)
def analyse_prompt(question: str = "something I want to find out in my data") -> str:
    """A one-command analysis coaching session.

    Args:
        question: what the student wants to find out. Defaults to a generic framing if they
            have not narrowed it down yet.
    """
    return (
        f"I want to analyse {question} using the GOAD method. Coach me through it as a "
        "mentor helping me think, not as someone doing the analysis for me.\n\n"
        "Step 1: call the `goad_analysis_checklist` tool with no stage to get the roadmap.\n"
        'Step 2: call it again with `stage="question"` and ask me those questions yourself, '
        "in the chat. Wait for my real answers — don't answer for me.\n"
        "Step 3: once I've answered, call the tool again with the same stage and a summary "
        "of what I said in `answers`, to record it and get the next stage. Repeat for every "
        "stage, in order — never skip ahead, and never fetch two stages before I've answered "
        "the first.\n"
        "Step 4: after the last stage, summarise what we worked out together. Only then, if "
        "I ask, write code — and tell me which stage each piece of it serves."
    )


@mcp.prompt(
    name="critique",
    description="Walk a finished visualisation through the course's critique checklist.",
)
def critique_prompt(plot: str = "the plot I just made") -> str:
    """A one-command critique session.

    Args:
        plot: short description of the visualisation being critiqued.
    """
    return (
        f"I want to critique {plot} against the GOAD checklist.\n\n"
        "Step 1: call the `goad_critique_visual` tool with no section to get the roadmap.\n"
        "Step 2: work through the four sections in order, one call each.\n"
        "Step 3: in sections 1 and 4 (first impression, and what I'm claiming), ask me and "
        "wait — you can't see the plot the way I do and you don't know what I'm claiming. In "
        "sections 2 and 3 (gestalt, the five guidelines), be specific and concrete: name the "
        "element, name the fix.\n"
        "Step 4: at the end, tell me plainly whether this is a result or still exploration."
    )


@mcp.prompt(
    name="distribution",
    description="Coach me through fitting and defending a distribution, in the right order.",
)
def distribution_prompt(variable: str = "a variable in my data") -> str:
    """A one-command distribution-fitting session.

    Args:
        variable: short description of the variable being fitted.
    """
    return (
        f"I want to work out how {variable} is distributed, using GOAD. Coach me through it "
        "in the right order.\n\n"
        "Step 1: call the `goad_distribution_method` tool with my variable to get the "
        "ordered checklist.\n"
        "Step 2: follow it with me in order. Before we fit anything, ask me what family I "
        "expect and why — a prediction I can be wrong about is what makes the fit worth "
        "doing.\n"
        "Step 3: do not let me decide what counts as an outlier until after step 8. If I "
        "reach for a z-score or an IQR rule before then, remind me what those assume.\n"
        "Step 4: call `goad_get_concept('05')` when we need the detail."
    )


# --------------------------------------------------------------------------- #
# Resources — each chapter is also addressable by URI
# --------------------------------------------------------------------------- #


@mcp.resource("goad://concept/{name}")
def concept_resource(name: str) -> str:
    """Expose a chapter's raw markdown as an MCP resource.

    `name` may be the chapter number, the doc filename ('04-five-families.md'), its stem, or
    the chapter title — the same lookup as goad_get_concept.
    """
    chapter = _find_chapter(name)
    if chapter is None:
        raise FileNotFoundError(f"No chapter named '{name}'")
    return _read_doc(chapter.file)


if __name__ == "__main__":
    mcp.run()
