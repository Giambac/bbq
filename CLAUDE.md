# BBQ — Back Branching Questioning

## What this project is

BBQ is an architecture for LLM-based task decomposition. It takes a complex (but clear, unambiguous) task and recursively breaks it into subtasks using a tree-search approach. The core insight is that the agent doesn't jump straight to subtasks — it first finds the *right question to ask* about a task, then answers that question to produce classified subtasks.

## Current state: Stage 3 (MCTS + execution)

Three implementations exist, each building on the last:

### Stage 1 — `bbq.py` (bare minimum decomposer)
Core 3-call loop (evaluate → question → decompose). BFS traversal, in-memory dicts. Tested and working with live API calls.

### Stage 2 — `bbq2.py` (feasibility + pruning)
Adds 4th LLM call for feasibility scoring (1-5), pruning (low feasibility + rule violations), SQLite persistence. Tested and working.

### Stage 3 — `bbq3.py` (MCTS + execution)
Replaces BFS with MCTS (UCB1 selection, LLM imagination rollouts, backpropagation). Adds:
- **Tool execution layer** — classifies leaf tasks by action type (code, web_search, file_io, human, reasoning) and dispatches to executors
- **Trial execution** — instead of just asking "can I do this?", tries doing it with a small budget
- **Resource budgets** — max wall-clock time, max API calls, max nodes
- **Solution-or-explanation contract** — returns either an actionable solution path or a structured unsatisfiability report
- **Human-in-the-loop** — prompts user interactively for tasks requiring human action

## Architecture decisions made

- **Three separate LLM calls per node** (not one big prompt) — keeps each call focused and small-context
- **BFS over DFS** for Stage 1 — gives breadth of options rather than drilling deep into one path
- **Minimal context per call** — each LLM call gets only the path from root to current node, not the full tree
- **Hard limits**: max depth 4, max children 5, max nodes 50
- **Subtask classification**: sufficient (alone achieves parent), necessary (must be done but not enough), co-sufficient (combined with others in same group = sufficient)
- **Sufficiency groups**: subtasks are grouped into combinations where each group is one complete path to achieving the parent

## Known criticalities (from design review)

These are the problems identified during the design phase that future stages need to address:

1. **LLMs can't reliably self-assess capability** — the evaluator call ("can I do this?") will sometimes hallucinate competence or be overly cautious. Stage 2 should replace or augment this with trial execution.
2. **Incomplete enumeration** — the questioning phase won't find *all* causal precursors. This is acceptable for a tool but means you can't trust "unsatisfiable" verdicts.
3. **Combinatorial explosion** — even with limits, real tasks produce large trees. A* was considered but requires a heuristic as hard as the problem. **Recommendation: use MCTS (Monte Carlo Tree Search) in Stage 3.**
4. **No executor** — Stage 1 is planning only. Stage 3 needs an action layer (API calls, code execution, human-in-the-loop).
5. **Context window pressure** — as trees grow, the "overall file" concept from the original design would blow up. **Decision: use SQLite in Stage 2** instead of files.
6. **Sufficient/necessary/co-sufficient classification is fuzzy** — the LLM makes judgment calls on every classification, and errors compound.

## Roadmap

### Stage 1 — DONE
- Core 3-call loop, BFS, in-memory, CLI

### Stage 2 — DONE
- Feasibility scoring, pruning, SQLite, enhanced JSON

### Stage 3 — DONE
- MCTS search, tool execution, resource budgets, solution path extraction

### Stage 4 (future)
- Real tool implementations (sandboxed code execution, web search API integration)
- Multi-model support (pluggable adapter pattern for different LLM providers)
- Parallel MCTS rollouts for faster search
- Learning from past decompositions (reuse subtree patterns)
- Web UI for interactive tree editing and execution monitoring

## Tech stack

- Python 3.10+
- `anthropic` SDK (only dependency)
- SQLite (Stage 2)
- No framework, no web server — CLI tool

## Running

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Stage 1 — simple decomposition
python bbq.py "Make me $1000 legally by today"

# Stage 2 — with feasibility scoring and pruning
python bbq2.py --min-feasibility 2 -o tree.json "Your complex task"

# Stage 3 — MCTS search with execution
python bbq3.py --budget-time 300 --mcts-iter 20 "Your complex task"
python bbq3.py --no-execute -o tree.json "Plan only, no execution"
python bbq3.py -v --db tree.db --viewer "Full run with viewer"
```

## File structure

```
bbq.py              — Stage 1 decomposer (3-call loop, in-memory, working + tested)
bbq2.py             — Stage 2 decomposer (4-call loop, SQLite, feasibility + pruning)
bbq3.py             — Stage 3 decomposer (MCTS, execution layer, resource budgets)
viewer.html         — Interactive tree visualizer (drag-drop JSON, dark/light mode)
README.md           — User-facing docs
CLAUDE.md           — This file (project context for Claude Code)
docs/
  original_idea.md  — Original BBQ architecture concept doc
  design_review.md  — Full analysis of criticalities and solutions
```

## Workflow rules

- **Commit and push frequently.** After completing any meaningful unit of work (a feature, a fix, a refactor), commit with a clean, descriptive message and push to GitHub. Never leave work uncommitted — we should never lose progress.
- Commit messages should be concise and describe *what changed and why*, not just "update files".
- Push to the remote after every commit so GitHub always reflects the latest state.

## Code style

- Single file for now (bbq.py), split when it exceeds ~500 lines
- Dataclasses for data structures, no Pydantic yet
- Type hints everywhere
- Prompts are inline strings (move to separate file if they grow)
- Temperature 0.3 for structured reasoning calls
