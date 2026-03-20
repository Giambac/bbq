# BBQ Stage 1 — Back Branching Questioning Decomposer

A task decomposition tool that breaks complex goals into actionable subtask trees using focused LLM calls.

## How it works

For each task node, BBQ makes **three separate LLM calls**:

1. **Evaluator** — "Can this task be done directly?" (yes → leaf node, done)
2. **Question finder** — "What question would reveal the causal precursors of this goal?"
3. **Decomposer** — Answers that question to produce classified subtasks

Subtasks are classified as:
- **Sufficient** — this alone achieves the parent
- **Necessary** — must be done, but not enough alone
- **Co-sufficient** — combined with others in the same group, they achieve the parent

The process repeats (BFS) until all nodes are leaves or limits are hit.

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Basic
python bbq.py "Make me $1000 legally by today"

# With options
python bbq.py --max-depth 3 --max-children 4 "Design a mobile app for elderly care"

# Save JSON output
python bbq.py -o tree.json "Plan a wedding for 200 guests in 2 months"

# Verbose (shows LLM call numbers)
python bbq.py -v "Build a SaaS product in one weekend"
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-depth` | 4 | Maximum tree depth |
| `--max-children` | 5 | Maximum subtasks per node |
| `--max-nodes` | 50 | Total node budget |
| `--model` | claude-sonnet-4-20250514 | Anthropic model to use |
| `-o FILE` | — | Save tree as JSON |
| `-v` | — | Verbose output |

## Output

The tool prints a formatted tree:

```
🌿 Make $1000 legally by today
├── ✅ ◆[sufficient|g1] Sell valuable possessions
├── 🌿 ◆[sufficient|g2] Do emergency freelance work
│   ├── ✅ ●[necessary|g1] Find urgent gig on Upwork
│   └── ✅ ●[necessary|g1] Complete deliverable within hours
└── ✅ ◆[sufficient|g3] Liquidate investments
```

## Next stages (not yet built)

- **Stage 2**: Add feasibility scoring (1-5), pruning, SQLite storage
- **Stage 3**: Search loop (MCTS), tool execution, resource budgets
