# BBQ Architecture — Design Review

This document captures the full critical analysis of the BBQ architecture performed before implementation began. It covers what's broken, what's good, and the specific engineering decisions made to address each problem.

## Critical problems identified

### 1. Self-assessment is unreliable
**Problem:** Step 2 ("Can I do this?") relies on the LLM evaluating its own capability. LLMs hallucinate competence or are overly cautious. The entire tree is built on this unreliable foundation.

**Mitigation (Stage 1):** Accept the limitation — Stage 1 is a planning tool, not an executor. The evaluator just needs to distinguish "single concrete action" from "multi-step task."

**Fix (Stage 3):** Replace self-assessment with trial execution. Try doing the task with a cheap, fast check. If the agent can write the code/call the API within a small token/time budget, it's satisfiable.

### 2. Incomplete causal enumeration
**Problem:** The questioning phase asks the LLM to find "all the causes" of a goal. LLMs generate plausible-sounding lists that miss critical paths and include irrelevant ones. You might declare something "unsatisfiable" when it isn't.

**Mitigation (Stage 1):** Constrain branching to 3-5 subtasks per node. Use structured decomposition templates in the question-finder prompt rather than free enumeration. Accept that completeness is impossible.

**Fix (Stage 2+):** Multiple decomposition passes from different angles. Cross-reference against known task taxonomies where applicable.

### 3. Combinatorial explosion
**Problem:** Each questioning step produces N subtasks, each producing N more. Exponential growth even with pruning. A* was proposed but requires a heuristic as hard as the problem itself.

**Mitigation (Stage 1):** Hard limits — max depth 4, max children 5, max nodes 50. BFS ensures breadth over depth.

**Fix (Stage 3):** Use MCTS (Monte Carlo Tree Search) instead of A*. MCTS doesn't need a heuristic — it uses random rollouts to estimate value. Have the LLM do quick "imagination rollouts" to guide search.

### 4. Information management paradox
**Problem:** The original design wants an "overall file" that's both small and complete. For non-trivial tasks, the relationship tree exceeds context window limits. Loading per-task files requires knowing which to load, which requires understanding the whole tree.

**Mitigation (Stage 1):** Keep it in-memory (Python dicts). Each LLM call gets only the path from root to current node — never the full tree.

**Fix (Stage 2):** Replace files with SQLite. A `nodes` table and `edges` table. The agent queries what it needs. This solves both context window limits and the "which file to load" problem.

### 5. Fuzzy classification
**Problem:** Sufficient/necessary/co-sufficient categories assume logical clarity that doesn't exist. "Get a freelance gig" — sufficient for $1000? Depends on the gig. Errors in classification compound through the tree.

**Mitigation (Stage 1):** Accept the fuzziness. The classification is a useful heuristic even if imprecise. Sufficiency groups help by making combinations explicit.

**Fix (Stage 2):** Add probability/feasibility scores (1-5). Replace binary classification with expected value estimates. Prune by score threshold rather than logical category.

### 6. No execution layer
**Problem:** The architecture is entirely planning. An agent needs to act — call APIs, write code, send messages. "Post on a freelancing platform" needs someone to actually do it.

**Mitigation (Stage 1):** Explicitly scope Stage 1 as a planning/decomposition tool only. The output is a prioritized plan for a human.

**Fix (Stage 3):** Each leaf node maps to a concrete action type (API call, code execution, human-in-the-loop request, information retrieval). Build an executor that dispatches based on action type.

### 7. Pruning is too binary
**Problem:** Pruning based on rule violations and "strict unsatisfiability" misses the real factors: cost, time, probability of success. Most failures are soft (unlikely), not hard (impossible).

**Mitigation (Stage 1):** No pruning in Stage 1 — just decompose everything within the node budget.

**Fix (Stage 2):** Add probability estimates to every branch. Select branches by expected value, not by logical classification. Set resource budgets (time, cost, API calls).

## What's genuinely good about the architecture

1. **The questioning phase is the key insight.** Most agent frameworks go straight from "task" to "subtasks." BBQ's step of first finding the *right question to ask* forces the LLM to think about the *structure* of the problem before proposing solutions. This is worth preserving and refining.

2. **Information management awareness.** The original design correctly identifies that multi-call LLM systems need careful context management. Most agent frameworks ignore this until they hit the wall.

3. **Solution-or-explanation contract.** Returning either a solution path or a structured explanation of unsatisfiability is a great UX pattern that most agent frameworks skip.

4. **Causal framing.** Asking "what causes this to be true?" rather than "what are the steps?" produces qualitatively different decompositions — more creative, less linear.

## Implementation decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM calls per node | 3 (evaluate, question, decompose) | Keeps each call focused and small-context |
| Search strategy (Stage 1) | BFS | Breadth-first gives useful output even if budget runs out early |
| Search strategy (Stage 3) | MCTS | No heuristic needed, fits LLM "imagination rollouts" |
| Storage (Stage 1) | In-memory dicts | Simplest possible, no dependencies |
| Storage (Stage 2+) | SQLite | Queryable, no context window issues, single-file |
| Context per LLM call | Root-to-node path only | Never dumps the full tree, keeps calls under ~500 tokens of context |
| API | Anthropic only (Stage 1) | Simpler. Pluggable adapter pattern for Stage 2+ if needed |
| Temperature | 0.3 | Low for structured reasoning, could experiment higher for question-finding |
