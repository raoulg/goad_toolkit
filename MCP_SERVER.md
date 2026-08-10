# GOAD🐐 as an MCP server

This repo ships a small [MCP](https://modelcontextprotocol.io) server,
[`goad_mcp.py`](goad_mcp.py), that turns the [`docs/`](docs) chapters into coaching methods
for a coding assistant (Claude Code, Cursor, Codex, Claude Desktop, …) — one for **analysing**
a dataset from a question rather than from its columns, one for **critiquing** a finished
visualisation, and one for **fitting a distribution** without deciding what an outlier is
first.

The point is to learn how to work *with* an assistant while doing data analysis, grounded in
GOAD's own method (the question comes first, the residual is the finding, every parameter is
part of the claim) instead of generic advice — and instead of handing the whole problem over
and getting back twelve plots nobody can defend.

## How to use it

- **Analysing something:** in Claude Code, run `/goad:analyse` (optionally describe what you
  want to find out). In any client, just say **"Coach me through analysing this with the goad
  server."** Expect a conversation, not an answer: your assistant will work through six
  stages, asking you questions one stage at a time and waiting for your actual answers before
  moving on.
- **Critiquing a plot:** `/goad:critique`, or **"Critique this plot against the goad
  checklist."**
- **Fitting a distribution:** `/goad:distribution`, or **"Coach me through fitting a
  distribution to this column using the goad server."**
- **Looking something up:** "Using the goad server, get the chapter on plot composition." /
  "Search the goad docs for long tail." / "What does goad say about t-SNE perplexity?"

The analysis and critique prompts are coaching sessions, not code generators — they are built
to ask what you think before offering an answer, and to write code only once you have said
what it is for. Using your assistant to answer all six stages itself in one go defeats the
point; ask it to go back to stage one and actually wait for your answers.

## What it exposes

**Tools**

| Tool | What it does |
| --- | --- |
| `goad_list_concepts` | List all 10 chapters, in reading order. |
| `goad_get_concept` | Fetch the full markdown of one chapter. |
| `goad_search` | Keyword search across every chapter. |
| `goad_analysis_checklist` | A six-stage, table-driven analysis conversation — see below. |
| `goad_critique_visual` | The visual-critique checklist, one section at a time. |
| `goad_distribution_method` | The method for fitting and defending a distribution, in the right order. |

**`goad_analysis_checklist` is the centrepiece**, and it is deliberately not a one-shot
document dump. It walks a student through six stages **in order**, one at a time, and only
advances once the assistant reports back what the student actually said:

1. **question** — what do you want to know, what would count as an answer in both directions,
   and who is it for?
2. **data** — what is one row, what was measured versus derived, and what does the question
   need that the data does not yet contain?
3. **shape** — how is each variable distributed, is anything long-tailed, and are the extreme
   values errors or ordinary members of a tail?
4. **encoding** — which of the five families does the question live in, and which visual
   channel carries the comparison you actually care about?
5. **critique** — what does the plot say in the first 200ms, what does the eye group by
   itself, and what exactly are you claiming?
6. **verification** — what null would produce this too, what confounder could, and what does
   the residual look like?

Each stage returns a markdown table — question / why it matters / what a good answer names —
plus an instruction to the assistant: ask the student, wait for a real answer, don't answer on
their behalf, and don't fetch the next stage until this one is actually recorded. Progress is
held in memory for the current session only. At the end it returns a recap of everything
recorded, which is a usable first draft of the write-up.

`goad_critique_visual` splits the work explicitly: sections 2 and 3 (gestalt principles, the
five guidelines) are ones an assistant can genuinely help with, and it is told to be concrete
there. Sections 1 and 4 (what caught your eye, what you are claiming) it is told to ask and
wait for — it cannot see the plot the way you do, and answering section 4 for you means
inventing a claim to fit the picture, which is backwards.

**Prompts**

- `analyse` — start the six-stage analysis conversation. Takes an optional `question`.
- `critique` — walk a finished visualisation through the checklist. Takes an optional `plot`.
- `distribution` — coach the distribution-fitting method in order. Takes an optional `variable`.

**Resources**

- `goad://concept/{name}` — each chapter's raw markdown, addressable by number (`5`),
  filename (`05-distributions.md`), stem, or title.

The content comes from [`docs/`](docs); no build step, no database, no embeddings.

## Requirements

- [`uv`](https://docs.astral.sh/uv/)
- Python 3.10+

You do **not** need `goad-toolkit` installed to run the server — it only reads markdown.

## Install

`uv` runs the server straight from a URL: it downloads the single script, provisions Python
and the `fastmcp` package from the script's own [PEP 723](https://peps.python.org/pep-0723/)
metadata, and the server fetches its docs from the repo over HTTP.

You pin a version with `GOAD_REF` — a git ref (branch, tag, or commit). That one variable
selects both the script (in the URL) and the docs it fetches (inside the script), so a whole
cohort runs exactly the same thing and the two cannot drift apart.

> **Note:** the docs and this server land in **v0.2.2**. 
> You can also point `GOAD_REF` to a branch or work from a local clone (see below).

### Claude Code

```bash
claude mcp add goad -e GOAD_REF=v0.2.2 -- \
  sh -c 'uv run --no-project https://raw.githubusercontent.com/raoulg/goad_toolkit/$GOAD_REF/goad_mcp.py'
```

### Cursor

Add to `.cursor/mcp.json` (per-project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "goad": {
      "command": "sh",
      "args": ["-c", "uv run --no-project https://raw.githubusercontent.com/raoulg/goad_toolkit/$GOAD_REF/goad_mcp.py"],
      "env": { "GOAD_REF": "v0.2.2" }
    }
  }
}
```

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.goad]
command = "sh"
args = ["-c", "uv run --no-project https://raw.githubusercontent.com/raoulg/goad_toolkit/$GOAD_REF/goad_mcp.py"]
env = { GOAD_REF = "v0.2.2" }
```

### Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config), then restart the
app:

```json
{
  "mcpServers": {
    "goad": {
      "command": "sh",
      "args": ["-c", "uv run --no-project https://raw.githubusercontent.com/raoulg/goad_toolkit/$GOAD_REF/goad_mcp.py"],
      "env": { "GOAD_REF": "v0.2.2" }
    }
  }
}
```

(The `sh` wrapper is for macOS/Linux; on Windows, drop `sh -c` and put the ref directly in
the URL instead, keeping the `GOAD_REF` env value in sync.)

## Working from a local clone instead

If you have already cloned `goad_toolkit`, point the same command at the local file instead
of the URL — the server then reads `docs/` straight off disk, so any edits show up without a
restart:

```bash
claude mcp add goad -- uv run --no-project /path/to/goad_toolkit/goad_mcp.py
```

## Tracking the bleeding edge instead

Set `GOAD_REF=main` to always run whatever is on `main`, including chapters or checklist
changes not in a release yet. Fine for your own use; for a cohort of students, pin to a tag
instead so everyone runs the exact same docs during an assignment.

## Update to a new version

Pick the newest tag from the [releases](https://github.com/raoulg/goad_toolkit/tags), then
point `GOAD_REF` at it. Because the ref lives in the URL, switching it fetches the new script
automatically — no cache clearing needed.

**Claude Code** — re-running `add` errors if the server already exists, so remove first, then
add with the new tag:

```bash
claude mcp remove goad -s user
claude mcp add goad -s user -e GOAD_REF=v0.2.2 -- \
  sh -c 'uv run --no-project https://raw.githubusercontent.com/raoulg/goad_toolkit/$GOAD_REF/goad_mcp.py'
```

**Cursor / Codex / Claude Desktop** — edit the `GOAD_REF` value in the config, then restart
your assistant so it relaunches the server.

## Companion servers

`goad` covers "is my analysis any good". Two siblings cover the rest of the same workflow:

- [`codestyle`](https://github.com/raoulg/codestyle) — is my code any good.
- [`vectormesh`](https://github.com/raoulg/vectormesh) — is my architecture any good.

All three follow the same shape: a `docs/` folder as the single source of truth, one PEP 723
script that serves it, and a staged conversation the assistant cannot short-circuit.
