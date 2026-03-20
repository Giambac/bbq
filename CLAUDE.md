# BBQ — Back Branching Questioning

## What this project is

BBQ is an architecture for LLM-based task decomposition. It takes a complex (but clear, unambiguous) task and recursively breaks it into subtasks using a tree-search approach. The core insight is that the agent doesn't jump straight to subtasks — it first finds the *right question to ask* about a task, then answers that question to produce classified subtasks.

## Current state: Stage 1 (bare minimum decomposer)

`bbq.py` is a working CLI tool that implements the core 3-call loop:
1. **Evaluator** — "Can this task be done directly?" → yes = leaf, no = continue
2. **Question finder** — "What question reveals the causal precursors?" → produces one well-formed question
3. **Decomposer** — Answers the question → produces subtasks classified as sufficient/necessary/co-sufficient

Uses BFS traversal, Anthropic API (Claude Sonnet), no database yet (in-memory dicts), outputs a tree to terminal + optional JSON export.

**It has NOT been tested with live API calls yet.** The data structures and tree output are verified working.

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

### Stage 2 (next)
- Add feasibility scoring (1-5 scale) to each node
- Add pruning: rule violations, impossibilities, low-feasibility cutoff
- Replace in-memory storage with SQLite (nodes + edges tables)
- JSON export with full metadata

### Stage 3 (later)
- Search loop with MCTS-style rollouts instead of pure BFS
- Tool execution layer (web search, code execution, file I/O)
- Resource budgets (max time, max cost, max API calls)
- Return either solution path or structured explanation of unsatisfiability

## Tech stack

- Python 3.10+
- `anthropic` SDK (only dependency)
- SQLite (Stage 2)
- No framework, no web server — CLI tool

## Running

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python bbq.py "Make me $1000 legally by today"
python bbq.py --max-depth 3 -o tree.json -v "Your complex task"
```

## File structure

```
bbq.py              — Main decomposer (Stage 1, working)
README.md           — User-facing docs
CLAUDE.md           — This file (project context for Claude Code)
docs/
  original_idea.md  — Original BBQ architecture concept doc
  design_review.md  — Full analysis of criticalities and solutions
```

## Code style

- Single file for now (bbq.py), split when it exceeds ~500 lines
- Dataclasses for data structures, no Pydantic yet
- Type hints everywhere
- Prompts are inline strings (move to separate file if they grow)
- Temperature 0.3 for structured reasoning calls
