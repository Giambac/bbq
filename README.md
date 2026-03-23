# BBQ — Back Branching Questioning

An LLM-powered task decomposition engine that breaks complex goals into actionable subtask trees. The core insight: instead of jumping straight to subtasks, BBQ first finds the *right question to ask* about a task, then answers that question to produce classified subtasks.

## The core idea

Most agent frameworks go straight from "task" to "subtasks." BBQ's questioning phase forces the LLM to think about the *structure* of the problem first by asking: "What question would reveal the causal precursors of this goal?" This produces qualitatively different decompositions — more creative, less linear.

Subtasks are classified by their relationship to the parent:
- **Sufficient** (◆) — this alone achieves the parent
- **Necessary** (●) — must be done, but not enough alone
- **Co-sufficient** (◐) — combined with others in the same group, they achieve the parent

Tasks are grouped into **sufficiency groups** — each group represents one complete path to achieving the parent goal.

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

Requires Python 3.10+ and an Anthropic API key.

---

## Stage 1 — `bbq.py` (Core Decomposer)

The bare minimum implementation. Takes a task and recursively decomposes it into a tree.

### How it works

For each node in the tree, three separate LLM calls are made:

1. **Evaluator** — "Can this task be done directly?" If yes, it becomes a leaf node. If no, continue.
2. **Question finder** — "What is the single best question whose answer reveals all the causal precursors of this goal?" Produces one well-formed question.
3. **Decomposer** — Answers that question to produce up to 5 classified subtasks with relationship types and sufficiency groups.

The tree is explored using **BFS** (breadth-first search), giving breadth of options rather than drilling deep into one path. Each LLM call receives only the path from root to current node — never the full tree — keeping context small.

### Features
- Three focused LLM calls per node (evaluate, question, decompose)
- BFS traversal
- In-memory storage (Python dicts)
- Terminal tree output with Unicode symbols
- JSON export
- Configurable limits: max depth, max children per node, max total nodes

### Usage

```bash
python bbq.py "Make me $1000 legally by today"
python bbq.py --max-depth 3 --max-children 4 "Design a mobile app for elderly care"
python bbq.py -v -o tree.json "Plan a wedding for 200 guests in 2 months"
python bbq.py --viewer "Auto-open interactive viewer"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-depth` | 4 | Maximum tree depth |
| `--max-children` | 5 | Maximum subtasks per node |
| `--max-nodes` | 50 | Total node budget |
| `--model` | claude-sonnet-4-20250514 | Anthropic model |
| `-o FILE` | — | Save tree as JSON |
| `-v` | — | Verbose output (shows LLM call numbers) |
| `--viewer` | — | Open interactive viewer in browser |

### Example output

```
🌿 Make $1000 legally by today
├── ◆[sufficient|g1] ✅ Sell valuable possessions online
├── 🌿 ◆[sufficient|g2] Do emergency freelance work
│   ├── ●[necessary|g1] ✅ Find urgent gig on Upwork/Fiverr
│   └── ●[necessary|g1] ✅ Complete deliverable within hours
└── ◆[sufficient|g3] ✅ Liquidate investments
```

### Limitations
- No feasibility assessment — all paths treated equally regardless of how realistic they are
- No pruning — every branch is explored within the node budget
- No execution — planning only
- In-memory storage — tree is lost when process exits

---

## Stage 2 — `bbq2.py` (Feasibility + Pruning)

Builds on Stage 1 by adding feasibility assessment and intelligent pruning. Stores everything in SQLite for persistence and queryability.

### How it works

Same three core LLM calls as Stage 1, plus two additional steps per child node:

4. **Feasibility scorer** — Rates each subtask on a 1-5 scale:
   - 5 = Straightforward, well-understood
   - 4 = Doable with effort
   - 3 = Uncertain, requires significant resources or luck
   - 2 = Unlikely, major obstacles
   - 1 = Near-impossible, technology doesn't exist

5. **Pruning checker** — Two checks per node:
   - **Feasibility cutoff** — prune anything below the minimum score (default: 2)
   - **Rule violation detection** — LLM checks if the subtask contradicts constraints in the root task (e.g., "steal money" is pruned when the root says "legally")

### Features
- Everything from Stage 1, plus:
- 4th LLM call for feasibility scoring (1-5 scale)
- Automatic pruning of low-feasibility and rule-violating branches
- SQLite persistence (nodes, edges, metadata tables)
- Enhanced JSON export with feasibility scores, pruning reasons, timing metadata
- Configurable feasibility threshold

### Usage

```bash
python bbq2.py "I want to become immortal"
python bbq2.py --min-feasibility 3 -o tree.json "Aggressive pruning example"
python bbq2.py --db tree.db -o tree.json "Save both SQLite and JSON"
python bbq2.py -v --viewer "Full run with viewer"
```

### Options

All Stage 1 options, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--min-feasibility` | 2 | Prune nodes below this score (1-5) |
| `--db FILE` | — | Save SQLite database to file |

### Example output

```
🌿 I want to become immortal.
├── 🌿 ◆[sufficient|g1] [██░░░] Achieve biological immortality through genetic engineering
│   ├── ⏳ ◐[co-sufficient|g1] [███░░] Develop gene therapy for telomerase activation
│   ├── ⏳ ◐[co-sufficient|g1] [██░░░] Engineer enhanced DNA repair mechanisms
│   └── ⏳ ●[necessary|g1] [██░░░] Develop delivery systems for genetic modifications
├── ✂  ◆[sufficient|g2] [█░░░░] Upload consciousness to digital substrate
│     ↳ feasibility too low (1/2 min)
├── ⏳ ◆[sufficient|g3] [██░░░] Develop nanotechnology for cellular repair
└── ⏳ ◆[sufficient|g5] [████░] Achieve philosophical immortality through legacy
```

Notice how consciousness upload (g2) was pruned at score 1/5, while philosophical immortality (g5) scored 4/5 — the feasibility scores immediately surface which paths are realistic.

### What Stage 2 solves
- **"All paths look equal"** → feasibility scores differentiate realistic from sci-fi
- **"Wasted API calls on dead ends"** → pruning cuts impossible branches early
- **"Tree lost after exit"** → SQLite persistence
- **"Contradictory subtasks"** → rule violation detection catches them

---

## Stage 3 — `bbq3.py` (MCTS + Execution)

The full architecture. Replaces BFS with Monte Carlo Tree Search for intelligent exploration, adds a tool execution layer, resource budgets, and returns either a solution path or a structured unsatisfiability report.

### How it works

**MCTS search loop** (replaces BFS):

1. **Select** — Walk the tree from root, picking the child with the highest UCB1 score at each level. UCB1 balances exploitation (high-value nodes) with exploration (unvisited nodes). Feasibility scores provide an additional boost.
2. **Expand** — Run the full pipeline on the selected node: evaluate → question → decompose → score feasibility → check pruning.
3. **Rollout** — Ask the LLM to "imagine" completing the task path quickly. Returns a success probability (0.0 to 1.0). Uses higher temperature (0.7) for creative simulation.
4. **Backpropagate** — Update visit counts and cumulative value from the rolled-out node all the way back to the root, so future selections are informed by past rollouts.

**Execution layer** — Each leaf node is classified by action type:

| Action type | What happens |
|---|---|
| `reasoning` | Auto-succeeds (pure thinking/analysis) |
| `human` | Prompts you interactively in the terminal to perform a physical action |
| `code` | Stub — will generate and run sandboxed Python (Stage 4) |
| `web_search` | Stub — will search the web via API (Stage 4) |
| `file_io` | Stub — will read/write files (Stage 4) |

**Trial execution** — Instead of just asking "can I do this?", Stage 3 actually tries to execute leaf tasks. If the executor succeeds, the node is marked `executed`. This addresses the key criticality that LLMs can't reliably self-assess capability.

**Solution contract** — The output is one of:
- **Solution path** — An ordered list of actionable leaf tasks forming the best complete path through the tree, with action types and execution status
- **Unsatisfiability report** — A structured explanation of why no viable path exists, listing pruned branches, low-feasibility nodes, and errors

### Features
- Everything from Stage 2, plus:
- MCTS search with UCB1 selection and LLM imagination rollouts
- Tool execution layer with 5 action types
- Trial execution replaces pure self-assessment
- Human-in-the-loop interactive prompts
- Resource budgets: max wall-clock time, max API calls, max nodes
- Solution path extraction from the tree
- Structured unsatisfiability report when no path works
- MCTS visit counts and rollout values on each node

### Usage

```bash
# Full run with execution (will prompt for human tasks)
python bbq3.py "Fix the leaking kitchen faucet today"

# Plan only, no execution
python bbq3.py --no-execute -o tree.json "Your complex task"

# Tuned MCTS search
python bbq3.py --mcts-iter 20 --budget-time 300 --budget-calls 100 "Your task"

# Full run with persistence and viewer
python bbq3.py -v --db tree.db --viewer "Your task"

# Quick test run with low budgets
python bbq3.py --max-nodes 10 --mcts-iter 3 --budget-calls 40 "Your task"
```

### Options

All Stage 1 and Stage 2 options, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--mcts-iter` | 30 | Number of MCTS search iterations |
| `--budget-time` | 600 | Max wall-clock seconds |
| `--budget-calls` | 200 | Max LLM API calls |
| `--no-execute` | — | Disable execution, plan only |

### Example output

```
🌿 Buy groceries for a dinner party tonight
├── ⚡ ●[necessary|g1] [█████] [reasoning] V=1 Create a detailed menu
├── 🌿 ●[necessary|g1] [█████] Determine number of guests attending
│   ├── ⏳ ◆[sufficient|g1] [█████] Call each guest directly
│   └── ⏳ ◆[sufficient|g2] [█████] Send text message to all guests
├── ⏳ ●[necessary|g1] [█████] Take inventory of existing ingredients
└── ⏳ ●[necessary|g1] [█████] Create shopping list with quantities

🎯 SOLUTION PATH
  ✅ 1. [reasoning] [█████] Create a detailed menu
      → Reasoning completed for: Create a detailed menu
  ○ 2. [reasoning] [█████] Call each guest directly
  ○ 3. [reasoning] [█████] Take inventory of existing ingredients
  ○ 4. [reasoning] [█████] Create shopping list with quantities
```

### What Stage 3 solves
- **"BFS explores everything equally"** → MCTS focuses on promising branches
- **"No way to act on the plan"** → execution layer dispatches to code/human/search
- **"Self-assessment is unreliable"** → trial execution proves capability instead of guessing
- **"No budget control"** → time, API call, and node budgets prevent runaway costs
- **"Just a tree, not a plan"** → solution path gives ordered actionable steps

---

## Interactive viewer

`viewer.html` is a self-contained HTML file that visualizes BBQ JSON output. No external dependencies.

Open it in any browser and drag-drop a JSON output file, or use the `--viewer` flag on any stage to auto-open it.

Features:
- Visual tree diagram with boxes connected by lines
- Color-coded nodes: green (leaf), orange (decomposed), red (pruned), gray (pending/max-depth), blue (executed)
- Relationship symbols (◆/●/◐) on edges
- Sufficiency group colors on node borders
- Feasibility score badges when present
- Click any node for full detail panel (task, evaluation reason, question asked, feasibility, pruning reason)
- Collapsible branches (depth 1 expanded by default)
- Expand all / Collapse all buttons
- Dark/light mode toggle
- Works with output from all three stages

## Project structure

```
bbq.py              — Stage 1: core 3-call decomposer (BFS, in-memory)
bbq2.py             — Stage 2: + feasibility scoring, pruning, SQLite
bbq3.py             — Stage 3: + MCTS search, execution layer, budgets
viewer.html         — Interactive tree visualizer (drag-drop JSON)
docs/
  original_idea.md  — Original BBQ architecture concept
  design_review.md  — Design criticalities and solutions
```

## Architecture

BBQ addresses several hard problems in LLM-based task decomposition:

| Problem | Stage 1 mitigation | Stage 2+ solution |
|---|---|---|
| Self-assessment unreliable | Accept it for planning | Trial execution (Stage 3) |
| Incomplete enumeration | Cap at 5 subtasks | Multiple angles, cross-reference |
| Combinatorial explosion | Hard depth/node limits | MCTS focuses search (Stage 3) |
| Context window pressure | Root-to-node path only | SQLite storage (Stage 2+) |
| Fuzzy classification | Accept as heuristic | Feasibility scores 1-5 (Stage 2+) |
| No execution | Planning tool only | Code/human/search executors (Stage 3) |

See [docs/design_review.md](docs/design_review.md) for the full analysis.

## Tech stack

- Python 3.10+
- `anthropic` SDK (only dependency)
- SQLite (Stage 2+)
- No framework, no web server — CLI tool
