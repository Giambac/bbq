#!/usr/bin/env python3
"""
BBQ Stage 3 — Back Branching Questioning with MCTS + Execution

Builds on Stage 2 with:
  - MCTS search loop (UCB1 selection, LLM rollouts, backpropagation)
  - Tool execution layer (code, web_search, file_io, human-in-the-loop)
  - Trial execution replaces self-assessment for leaf detection
  - Resource budgets (max time, max cost estimate, max API calls)
  - Solution path extraction or structured unsatisfiability report

Usage:
  python bbq3.py "Make me $1000 legally by today"
  python bbq3.py --budget-time 300 --budget-calls 100 "Your complex task"
  python bbq3.py --no-execute -o tree.json "Plan only, no execution"
  python bbq3.py -v --db tree.db --viewer "Full run with persistence"
"""

import argparse
import json
import math
import sqlite3
import subprocess
import sys
import os
import tempfile
import time

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from typing import Optional
from anthropic import Anthropic

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_DEPTH = 5
MAX_CHILDREN = 5
MAX_NODES = 50
MIN_FEASIBILITY = 2
MCTS_ITERATIONS = 30       # number of MCTS iterations before stopping
UCB_EXPLORATION = 1.41     # exploration constant (sqrt(2) is standard)
BUDGET_TIME = 600           # max seconds (10 min default)
BUDGET_CALLS = 200          # max LLM API calls
ROLLOUT_TEMPERATURE = 0.7   # higher temp for creative rollouts

# ── Action types ─────────────────────────────────────────────────────────────

ACTION_TYPES = ["code", "web_search", "file_io", "human", "reasoning"]

# ── SQLite storage ──────────────────────────────────────────────────────────

class TreeDB:
    """SQLite-backed storage for the decomposition tree."""

    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id                INTEGER PRIMARY KEY,
                task              TEXT NOT NULL,
                depth             INTEGER NOT NULL DEFAULT 0,
                status            TEXT NOT NULL DEFAULT 'pending',
                can_do_directly   INTEGER,
                evaluation_reason TEXT DEFAULT '',
                question_asked    TEXT DEFAULT '',
                relationship      TEXT DEFAULT '',
                group_id          INTEGER DEFAULT 0,
                feasibility       INTEGER,
                feasibility_reason TEXT DEFAULT '',
                pruned_reason     TEXT DEFAULT '',
                action_type       TEXT DEFAULT '',
                execution_result  TEXT DEFAULT '',
                execution_success INTEGER,
                rollout_value     REAL DEFAULT 0.0,
                visits            INTEGER DEFAULT 0,
                created_at        REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
                parent_id INTEGER NOT NULL,
                child_id  INTEGER NOT NULL,
                PRIMARY KEY (parent_id, child_id),
                FOREIGN KEY (parent_id) REFERENCES nodes(id),
                FOREIGN KEY (child_id)  REFERENCES nodes(id)
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self.conn.commit()

    def insert_node(self, node_id: int, task: str, depth: int,
                    relationship: str = "", group_id: int = 0) -> None:
        self.conn.execute(
            "INSERT INTO nodes (id, task, depth, relationship, group_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (node_id, task, depth, relationship, group_id, time.time())
        )
        self.conn.commit()

    def insert_edge(self, parent_id: int, child_id: int) -> None:
        self.conn.execute(
            "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id)
        )
        self.conn.commit()

    def update_node(self, node_id: int, **fields) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [node_id]
        self.conn.execute(f"UPDATE nodes SET {sets} WHERE id = ?", vals)
        self.conn.commit()

    def get_node(self, node_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()

    def get_children(self, node_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT n.* FROM nodes n JOIN edges e ON n.id = e.child_id "
            "WHERE e.parent_id = ? ORDER BY n.id", (node_id,)
        ).fetchall()

    def get_parent_id(self, node_id: int) -> Optional[int]:
        row = self.conn.execute(
            "SELECT parent_id FROM edges WHERE child_id = ?", (node_id,)
        ).fetchone()
        return row["parent_id"] if row else None

    def count_nodes(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def get_expandable_nodes(self) -> list[sqlite3.Row]:
        """Get nodes that can still be expanded (pending, not pruned, not max-depth)."""
        return self.conn.execute(
            "SELECT * FROM nodes WHERE status = 'pending'"
        ).fetchall()

    def get_all_nodes(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()

    def set_metadata(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value)
        )
        self.conn.commit()

    def get_all_metadata(self) -> dict:
        rows = self.conn.execute("SELECT key, value FROM metadata").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def close(self):
        self.conn.close()


# ── Executor ─────────────────────────────────────────────────────────────────

class Executor:
    """Executes leaf node tasks based on their action type."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def execute(self, task: str, action_type: str) -> tuple[bool, str]:
        """Execute a task. Returns (success, result_description)."""
        dispatch = {
            "code": self._execute_code,
            "web_search": self._execute_web_search,
            "file_io": self._execute_file_io,
            "human": self._execute_human,
            "reasoning": self._execute_reasoning,
        }
        handler = dispatch.get(action_type, self._execute_reasoning)
        return handler(task)

    def _execute_code(self, task: str) -> tuple[bool, str]:
        """Generate and run Python code for the task."""
        # For safety, we just report what would be done
        # Real implementation would use a sandboxed subprocess
        return False, f"[code execution stub] Would generate and run Python code for: {task[:100]}"

    def _execute_web_search(self, task: str) -> tuple[bool, str]:
        """Search the web for information."""
        return False, f"[web search stub] Would search for: {task[:100]}"

    def _execute_file_io(self, task: str) -> tuple[bool, str]:
        """Read or write files."""
        return False, f"[file I/O stub] Would perform file operation for: {task[:100]}"

    def _execute_human(self, task: str) -> tuple[bool, str]:
        """Request human action via interactive prompt."""
        print(f"\n{'─'*50}")
        print(f"🙋 HUMAN ACTION REQUIRED:")
        print(f"   {task}")
        print(f"{'─'*50}")
        try:
            response = input("Did you complete this? (yes/no/skip): ").strip().lower()
            if response in ("yes", "y"):
                detail = input("Brief description of what you did (or Enter to skip): ").strip()
                return True, detail or "Completed by human"
            elif response in ("skip", "s"):
                return False, "Skipped by human"
            else:
                return False, "Human declined to complete task"
        except (EOFError, KeyboardInterrupt):
            return False, "Human input not available"

    def _execute_reasoning(self, task: str) -> tuple[bool, str]:
        """Pure reasoning task — always succeeds as it's just thinking."""
        return True, f"Reasoning completed for: {task[:100]}"


# ── MCTS Decomposer ─────────────────────────────────────────────────────────

class BBQDecomposer3:
    def __init__(self, model: str = DEFAULT_MODEL, max_depth: int = MAX_DEPTH,
                 max_children: int = MAX_CHILDREN, max_nodes: int = MAX_NODES,
                 min_feasibility: int = MIN_FEASIBILITY,
                 mcts_iterations: int = MCTS_ITERATIONS,
                 budget_time: int = BUDGET_TIME, budget_calls: int = BUDGET_CALLS,
                 execute: bool = True,
                 verbose: bool = False, db_path: str = ":memory:"):
        self.client = Anthropic()
        self.model = model
        self.max_depth = max_depth
        self.max_children = max_children
        self.max_nodes = max_nodes
        self.min_feasibility = min_feasibility
        self.mcts_iterations = mcts_iterations
        self.budget_time = budget_time
        self.budget_calls = budget_calls
        self.execute_enabled = execute
        self.verbose = verbose
        self.db = TreeDB(db_path)
        self.executor = Executor(verbose=verbose)
        self.next_id = 0
        self.total_calls = 0
        self.total_pruned = 0
        self.total_executed = 0
        self.total_rollouts = 0
        self.start_time = 0.0
        self.root_task = ""

    # ── Budget checks ────────────────────────────────────────────────────

    def _budget_remaining(self) -> bool:
        """Check if we still have budget to continue."""
        if self.total_calls >= self.budget_calls:
            if self.verbose:
                print(f"  ⚠ API call budget exhausted ({self.budget_calls})")
            return False
        if time.time() - self.start_time >= self.budget_time:
            if self.verbose:
                print(f"  ⚠ Time budget exhausted ({self.budget_time}s)")
            return False
        if self.db.count_nodes() >= self.max_nodes:
            if self.verbose:
                print(f"  ⚠ Node budget exhausted ({self.max_nodes})")
            return False
        return True

    # ── LLM interface ────────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str, max_tokens: int = 1024,
                  temperature: float = 0.3) -> str:
        """Single LLM call with budget tracking."""
        if not self._budget_remaining():
            raise BudgetExhausted("No budget remaining")
        self.total_calls += 1
        if self.verbose:
            print(f"  [LLM call #{self.total_calls}]")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return response.content[0].text

    # ── Core LLM calls (evaluate, question, decompose, score, prune) ────

    def _evaluate(self, node_id: int, path_context: str) -> tuple[bool, str]:
        """Call 1: Can this task be done directly?"""
        node = self.db.get_node(node_id)
        system = (
            "You are an evaluator for a task decomposition system. "
            "You assess whether a task can be accomplished DIRECTLY in a single step "
            "by an LLM agent (with access to tools: code execution, web search, file I/O) "
            "or by a human performing one concrete action.\n\n"
            "A task is 'directly doable' if it requires no further breakdown — it's a single, "
            "concrete action. Complex multi-step tasks are NOT directly doable.\n\n"
            "Respond in exactly this format:\n"
            "VERDICT: YES or NO\n"
            "REASON: one sentence explaining why"
        )
        user = f"Task context (path from root):\n{path_context}\n\nTask to evaluate:\n{node['task']}"
        result = self._call_llm(system, user)
        verdict = "YES" in result.split("REASON")[0].upper()
        reason = result.split("REASON:")[-1].strip() if "REASON:" in result else result
        return verdict, reason

    def _find_question(self, node_id: int, path_context: str) -> str:
        """Call 2: What question reveals causal precursors?"""
        node = self.db.get_node(node_id)
        system = (
            "You are a strategic question designer for a task decomposition system called BBQ "
            "(Back Branching Questioning). Find the SINGLE BEST question whose answer reveals "
            "all the causal precursors for the given task.\n\n"
            "The question should:\n"
            "- Target the CAUSES and COMPONENTS of the task\n"
            "- Be specific enough to yield actionable subtasks\n"
            "- Be broad enough to not miss important paths\n"
            "- Frame things in terms of what a person/agent could actually do\n\n"
            "Respond with ONLY the question, nothing else."
        )
        user = f"Task context (path from root):\n{path_context}\n\nTask to decompose:\n{node['task']}"
        return self._call_llm(system, user).strip().strip('"')

    def _decompose(self, node_id: int, question: str, path_context: str) -> list[dict]:
        """Call 3: Answer the question to produce classified subtasks."""
        node = self.db.get_node(node_id)
        system = (
            "You are a task decomposer for the BBQ (Back Branching Questioning) system. "
            "You are given a task and a guiding question. Answer the question to produce "
            f"concrete subtasks (maximum {self.max_children}).\n\n"
            "For each subtask, classify its relationship to the parent task:\n"
            "- SUFFICIENT: this subtask alone would achieve the parent task\n"
            "- NECESSARY: this must be done, but alone it's not enough\n"
            "- CO-SUFFICIENT: combined with other co-sufficient tasks in the same group, "
            "they together are sufficient\n\n"
            "Group tasks into GROUPS. Each group = one complete path to the parent.\n\n"
            "Respond in this exact JSON format (no markdown, no extra text):\n"
            "[\n"
            '  {"description": "...", "relationship": "sufficient|necessary|co-sufficient", "group": 1},\n'
            '  {"description": "...", "relationship": "necessary", "group": 2},\n'
            '  {"description": "...", "relationship": "co-sufficient", "group": 2}\n'
            "]"
        )
        user = (
            f"Task context (path from root):\n{path_context}\n\n"
            f"Task to decompose:\n{node['task']}\n\n"
            f"Guiding question:\n{question}"
        )
        result = self._call_llm(system, user, max_tokens=2048)

        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1
            if start >= 0 and end > start:
                items = json.loads(cleaned[start:end])
            else:
                raise ValueError(f"Could not parse decomposer output:\n{result}")

        return [
            {
                "description": item["description"],
                "relationship": item.get("relationship", "sufficient"),
                "group": item.get("group", 1),
            }
            for item in items[:self.max_children]
        ]

    def _score_feasibility(self, node_id: int, path_context: str) -> tuple[int, str]:
        """Call 4: Rate feasibility 1-5."""
        node = self.db.get_node(node_id)
        system = (
            "You are a feasibility assessor for a task decomposition system. "
            "Rate how feasible the given task is on a 1-5 scale:\n\n"
            "5 = Straightforward — well-understood, can be done with standard methods\n"
            "4 = Doable with effort — requires skill or resources but clearly achievable\n"
            "3 = Uncertain — requires significant research, resources, or luck\n"
            "2 = Unlikely — major obstacles, would require breakthroughs\n"
            "1 = Near-impossible — technology doesn't exist, violates known constraints\n\n"
            "Respond in exactly this format:\n"
            "SCORE: <number 1-5>\n"
            "REASON: one sentence explaining the rating"
        )
        user = f"Task context (path from root):\n{path_context}\n\nTask to assess:\n{node['task']}"
        result = self._call_llm(system, user)

        score = 3
        for line in result.splitlines():
            if line.strip().upper().startswith("SCORE"):
                for ch in line:
                    if ch.isdigit() and 1 <= int(ch) <= 5:
                        score = int(ch)
                        break
                break
        reason = result.split("REASON:")[-1].strip() if "REASON:" in result else result
        return score, reason

    def _check_pruning(self, node_id: int, path_context: str) -> tuple[bool, str]:
        """Check if a node should be pruned."""
        node = self.db.get_node(node_id)

        if node["feasibility"] is not None and node["feasibility"] < self.min_feasibility:
            return True, f"feasibility too low ({node['feasibility']}/{self.min_feasibility} min)"

        system = (
            "You are a constraint checker. Check whether a subtask CONTRADICTS or VIOLATES "
            "any constraints implied by the root task.\n\n"
            "Respond in exactly this format:\n"
            "VERDICT: VALID or VIOLATION\n"
            "REASON: one sentence explaining why"
        )
        user = (
            f"Root task:\n{self.root_task}\n\n"
            f"Task context:\n{path_context}\n\n"
            f"Subtask to check:\n{node['task']}"
        )
        result = self._call_llm(system, user)
        is_violation = "VIOLATION" in result.split("REASON")[0].upper()
        reason = result.split("REASON:")[-1].strip() if "REASON:" in result else result
        if is_violation:
            return True, f"rule violation: {reason}"
        return False, ""

    # ── Stage 3: Action type classification ──────────────────────────────

    def _classify_action(self, node_id: int, path_context: str) -> str:
        """Classify what type of action a leaf task requires."""
        node = self.db.get_node(node_id)
        system = (
            "You classify tasks by what type of action they require. Choose EXACTLY ONE:\n\n"
            "- code: requires writing and running code (scripts, data processing, API calls)\n"
            "- web_search: requires searching the internet for information\n"
            "- file_io: requires reading or writing files/documents\n"
            "- human: requires a human to physically do something (buy, call, meet, build)\n"
            "- reasoning: requires only thinking/analysis, no external tools needed\n\n"
            "Respond with ONLY the action type, nothing else."
        )
        user = f"Task context:\n{path_context}\n\nTask:\n{node['task']}"
        result = self._call_llm(system, user, max_tokens=32).strip().lower()

        # Match to valid action type
        for at in ACTION_TYPES:
            if at in result:
                return at
        return "reasoning"

    # ── Stage 3: MCTS rollout ────────────────────────────────────────────

    def _rollout(self, node_id: int, path_context: str) -> float:
        """Imagination rollout: LLM quickly imagines completing this path.
        Returns estimated success probability (0.0 to 1.0)."""
        node = self.db.get_node(node_id)
        self.total_rollouts += 1
        system = (
            "You are simulating whether a task path can succeed. Given the task and its "
            "context, imagine quickly trying to complete it. Consider:\n"
            "- Are there obvious blockers or impossibilities?\n"
            "- Does the agent/person have the necessary capabilities?\n"
            "- Is the timeline realistic?\n"
            "- Are there missing prerequisites?\n\n"
            "Respond in exactly this format:\n"
            "SUCCESS_PROBABILITY: <number 0.0 to 1.0>\n"
            "REASONING: one sentence"
        )
        user = f"Root goal:\n{self.root_task}\n\nCurrent path:\n{path_context}\n\nTask to simulate:\n{node['task']}"

        result = self._call_llm(system, user, max_tokens=256, temperature=ROLLOUT_TEMPERATURE)

        prob = 0.5  # default
        for line in result.splitlines():
            if "SUCCESS_PROBABILITY" in line.upper():
                for token in line.split():
                    try:
                        val = float(token)
                        if 0.0 <= val <= 1.0:
                            prob = val
                            break
                    except ValueError:
                        continue
                break
        return prob

    # ── MCTS core ────────────────────────────────────────────────────────

    def _ucb1_score(self, node_id: int, parent_visits: int) -> float:
        """UCB1 score for node selection."""
        node = self.db.get_node(node_id)
        visits = node["visits"] or 0
        value = node["rollout_value"] or 0.0

        if visits == 0:
            return float('inf')  # always explore unvisited nodes first

        exploitation = value / visits
        exploration = UCB_EXPLORATION * math.sqrt(math.log(parent_visits) / visits)

        # Boost from feasibility score if available
        feas_bonus = 0.0
        if node["feasibility"] is not None:
            feas_bonus = (node["feasibility"] - 1) / 8.0  # 0.0 to 0.5 bonus

        return exploitation + exploration + feas_bonus

    def _select(self, root_id: int) -> int:
        """MCTS selection: walk tree picking best UCB1 child until expandable node."""
        current_id = root_id
        while True:
            node = self.db.get_node(current_id)
            children = self.db.get_children(current_id)

            # If node has no children or is a terminal state, return it
            if not children or node["status"] in ("leaf", "pruned", "executed", "error"):
                return current_id

            # Check if there are unexpanded children (pending status)
            pending = [c for c in children if c["status"] == "pending"]
            if pending:
                # Return the first unexpanded child
                return pending[0]["id"]

            # All children expanded — pick by UCB1
            parent_visits = node["visits"] or 1
            active_children = [c for c in children if c["status"] not in ("pruned",)]
            if not active_children:
                return current_id

            best_id = max(active_children, key=lambda c: self._ucb1_score(c["id"], parent_visits))["id"]
            current_id = best_id

    def _backpropagate(self, node_id: int, value: float):
        """Propagate rollout value up the tree."""
        current_id = node_id
        while current_id is not None:
            node = self.db.get_node(current_id)
            new_visits = (node["visits"] or 0) + 1
            new_value = (node["rollout_value"] or 0.0) + value
            self.db.update_node(current_id, visits=new_visits, rollout_value=new_value)
            current_id = self.db.get_parent_id(current_id)

    # ── Tree operations ──────────────────────────────────────────────────

    def _create_node(self, task: str, parent_id: Optional[int] = None,
                     relationship: str = "", group: int = 0) -> int:
        node_id = self.next_id
        depth = 0
        if parent_id is not None:
            parent = self.db.get_node(parent_id)
            depth = parent["depth"] + 1
        self.db.insert_node(node_id, task, depth, relationship, group)
        if parent_id is not None:
            self.db.insert_edge(parent_id, node_id)
        self.next_id += 1
        return node_id

    def _get_path_context(self, node_id: int) -> str:
        """Build minimal context: task descriptions from root to this node."""
        path = []
        current_id = node_id
        while current_id is not None:
            node = self.db.get_node(current_id)
            path.append(node["task"])
            current_id = self.db.get_parent_id(current_id)
        path.reverse()
        return " → ".join(path)

    # ── Expand a node (evaluate → question → decompose → score) ──────────

    def _expand_node(self, node_id: int) -> list[int]:
        """Run the full expansion on a node. Returns list of new child IDs."""
        node = self.db.get_node(node_id)
        path_context = self._get_path_context(node_id)
        indent = "  " * node["depth"]
        new_children = []

        if node["depth"] >= self.max_depth:
            self.db.update_node(node_id, status="max-depth")
            if self.verbose:
                print(f"{indent}⚠ Max depth reached: {node['task'][:50]}...")
            return []

        # ── Call 1: Evaluate ──
        print(f"{indent}📋 Evaluating: {node['task'][:60]}{'...' if len(node['task']) > 60 else ''}")
        can_do, reason = self._evaluate(node_id, path_context)
        self.db.update_node(node_id,
                            can_do_directly=1 if can_do else 0,
                            evaluation_reason=reason)

        if can_do:
            # ── Classify action type ──
            action_type = self._classify_action(node_id, path_context)
            self.db.update_node(node_id, status="leaf", action_type=action_type)
            print(f"{indent}   ✅ Leaf [{action_type}] — {reason[:50]}")

            # ── Trial execution (Stage 3 key feature) ──
            if self.execute_enabled and action_type != "human":
                print(f"{indent}   ⚡ Attempting execution...")
                success, exec_result = self.executor.execute(node["task"], action_type)
                self.db.update_node(node_id,
                                    status="executed" if success else "leaf",
                                    execution_result=exec_result,
                                    execution_success=1 if success else 0)
                self.total_executed += 1
                if success:
                    print(f"{indent}   ✅ Executed — {exec_result[:50]}")
                else:
                    print(f"{indent}   ⏸ Not executed — {exec_result[:50]}")
            return []

        print(f"{indent}   ❌ Cannot do directly — {reason[:60]}")

        # ── Call 2: Find question ──
        print(f"{indent}   🔍 Finding decomposition question...")
        question = self._find_question(node_id, path_context)
        self.db.update_node(node_id, question_asked=question)
        print(f"{indent}   ❓ \"{question[:70]}{'...' if len(question) > 70 else ''}\"")

        # ── Call 3: Decompose ──
        print(f"{indent}   🌿 Decomposing...")
        try:
            subtasks = self._decompose(node_id, question, path_context)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"{indent}   ⚠ Decomposition failed: {e}")
            self.db.update_node(node_id, status="error")
            return []

        self.db.update_node(node_id, status="decomposed")

        for st in subtasks:
            if not self._budget_remaining():
                break

            child_id = self._create_node(
                task=st["description"],
                parent_id=node_id,
                relationship=st["relationship"],
                group=st["group"],
            )
            child_path = self._get_path_context(child_id)

            # ── Call 4: Feasibility score ──
            score, score_reason = self._score_feasibility(child_id, child_path)
            self.db.update_node(child_id, feasibility=score, feasibility_reason=score_reason)

            score_bar = "█" * score + "░" * (5 - score)
            rel_symbol = {"sufficient": "◆", "necessary": "●", "co-sufficient": "◐"}.get(st["relationship"], "?")
            print(f"{indent}   {rel_symbol} [{st['relationship']}|g{st['group']}] [{score_bar}] {st['description'][:45]}")

            # ── Pruning check ──
            should_prune, prune_reason = self._check_pruning(child_id, child_path)
            if should_prune:
                self.db.update_node(child_id, status="pruned", pruned_reason=prune_reason)
                self.total_pruned += 1
                print(f"{indent}      ✂ PRUNED — {prune_reason[:55]}")
            else:
                new_children.append(child_id)

        print()
        return new_children

    # ── Main MCTS loop ───────────────────────────────────────────────────

    def decompose(self, task: str) -> dict:
        """Run the full BBQ Stage 3 MCTS decomposition."""
        self.start_time = time.time()
        self.root_task = task

        print(f"\n{'='*60}")
        print(f"BBQ Decomposer — Stage 3 (MCTS)")
        print(f"{'='*60}")
        print(f"Task: {task}")
        print(f"Limits: depth={self.max_depth}, children={self.max_children}, nodes={self.max_nodes}")
        print(f"Budget: {self.budget_time}s, {self.budget_calls} calls, {self.mcts_iterations} iterations")
        print(f"Feasibility cutoff: {self.min_feasibility}/5")
        print(f"Execution: {'enabled' if self.execute_enabled else 'disabled (plan only)'}")
        print(f"{'='*60}\n")

        # Save metadata
        self.db.set_metadata("root_task", task)
        self.db.set_metadata("model", self.model)
        self.db.set_metadata("stage", "3")
        self.db.set_metadata("search_strategy", "MCTS")
        self.db.set_metadata("max_depth", str(self.max_depth))
        self.db.set_metadata("max_children", str(self.max_children))
        self.db.set_metadata("max_nodes", str(self.max_nodes))
        self.db.set_metadata("min_feasibility", str(self.min_feasibility))
        self.db.set_metadata("mcts_iterations", str(self.mcts_iterations))
        self.db.set_metadata("budget_time", str(self.budget_time))
        self.db.set_metadata("budget_calls", str(self.budget_calls))
        self.db.set_metadata("execution_enabled", str(self.execute_enabled))
        self.db.set_metadata("start_time", str(self.start_time))

        # Create root and do initial expansion
        root_id = self._create_node(task)

        try:
            self._expand_node(root_id)
        except BudgetExhausted:
            print("⚠ Budget exhausted during initial expansion")

        # ── MCTS iterations ──
        iteration = 0
        while iteration < self.mcts_iterations and self._budget_remaining():
            iteration += 1

            expandable = self.db.get_expandable_nodes()
            if not expandable:
                if self.verbose:
                    print(f"  [MCTS] No more expandable nodes after {iteration} iterations")
                break

            if self.verbose:
                print(f"\n{'─'*40} MCTS iteration {iteration}/{self.mcts_iterations} {'─'*10}")

            # 1. SELECT — pick most promising unexpanded node
            selected_id = self._select(root_id)
            selected = self.db.get_node(selected_id)

            if selected["status"] != "pending":
                # Already expanded, do a rollout and backpropagate
                path_context = self._get_path_context(selected_id)
                try:
                    rollout_value = self._rollout(selected_id, path_context)
                    self._backpropagate(selected_id, rollout_value)
                except BudgetExhausted:
                    break
                continue

            # 2. EXPAND — run evaluate → question → decompose
            try:
                new_children = self._expand_node(selected_id)
            except BudgetExhausted:
                print("⚠ Budget exhausted during expansion")
                break

            # 3. ROLLOUT — simulate from one new child (or the node itself)
            rollout_target = new_children[0] if new_children else selected_id
            try:
                path_context = self._get_path_context(rollout_target)
                rollout_value = self._rollout(rollout_target, path_context)
            except BudgetExhausted:
                break

            # 4. BACKPROPAGATE — update scores up the tree
            self._backpropagate(rollout_target, rollout_value)

        # ── Execute human tasks if enabled ──
        if self.execute_enabled:
            self._execute_human_tasks()

        elapsed = time.time() - self.start_time
        self.db.set_metadata("end_time", str(time.time()))
        self.db.set_metadata("total_calls", str(self.total_calls))
        self.db.set_metadata("total_nodes", str(self.db.count_nodes()))
        self.db.set_metadata("total_pruned", str(self.total_pruned))
        self.db.set_metadata("total_executed", str(self.total_executed))
        self.db.set_metadata("total_rollouts", str(self.total_rollouts))
        self.db.set_metadata("mcts_iterations_completed", str(iteration))
        self.db.set_metadata("elapsed_seconds", f"{elapsed:.1f}")

        print(f"\n{'='*60}")
        print(f"Done. {self.db.count_nodes()} nodes, {self.total_calls} LLM calls, "
              f"{self.total_pruned} pruned, {self.total_rollouts} rollouts, {elapsed:.1f}s")
        print(f"{'='*60}\n")

        return self._build_output()

    def _execute_human_tasks(self):
        """Prompt the user for any leaf tasks classified as 'human'."""
        nodes = self.db.get_all_nodes()
        human_leaves = [n for n in nodes
                        if n["status"] == "leaf" and n["action_type"] == "human"]
        if not human_leaves:
            return

        print(f"\n{'='*60}")
        print(f"🙋 HUMAN TASKS ({len(human_leaves)} remaining)")
        print(f"{'='*60}")

        for node in human_leaves:
            success, result = self.executor.execute(node["task"], "human")
            self.db.update_node(node["id"],
                                status="executed" if success else "leaf",
                                execution_result=result,
                                execution_success=1 if success else 0)
            self.total_executed += 1

    # ── Solution path extraction ─────────────────────────────────────────

    def _extract_solution_path(self) -> Optional[list[dict]]:
        """Find the best complete solution path from root to leaves.
        Returns ordered list of actionable tasks, or None if no path found."""

        def _score_group(parent_id: int, group_id: int) -> tuple[float, list[dict]]:
            """Score a sufficiency group. Returns (score, ordered_tasks)."""
            children = self.db.get_children(parent_id)
            group_nodes = [c for c in children if c["group_id"] == group_id]
            if not group_nodes:
                return 0.0, []

            tasks = []
            total_score = 0.0
            all_resolved = True

            for node in group_nodes:
                if node["status"] == "pruned":
                    return 0.0, []  # group is broken if any member is pruned

                node_score = (node["feasibility"] or 3) / 5.0
                if node["visits"] and node["visits"] > 0:
                    node_score = (node_score + node["rollout_value"] / node["visits"]) / 2

                sub_children = self.db.get_children(node["id"])
                if sub_children:
                    # Recursively find best group in subtree
                    best_sub_score = 0.0
                    best_sub_tasks = []
                    groups = set(c["group_id"] for c in sub_children)
                    for g in groups:
                        gs, gt = _score_group(node["id"], g)
                        if gs > best_sub_score:
                            best_sub_score = gs
                            best_sub_tasks = gt
                    node_score *= best_sub_score if best_sub_score > 0 else 0.5
                    tasks.extend(best_sub_tasks)
                elif node["status"] in ("leaf", "executed", "pending"):
                    tasks.append({
                        "id": node["id"],
                        "task": node["task"],
                        "action_type": node["action_type"] or "reasoning",
                        "feasibility": node["feasibility"],
                        "status": node["status"],
                        "execution_result": node["execution_result"] or "",
                    })
                    if node["status"] == "pending":
                        all_resolved = False

                total_score += node_score

            avg_score = total_score / len(group_nodes) if group_nodes else 0.0
            return avg_score, tasks

        # Find the best group at root level
        root_children = self.db.get_children(0)
        if not root_children:
            return None

        groups = set(c["group_id"] for c in root_children)
        best_score = 0.0
        best_path = None

        for g in groups:
            score, tasks = _score_group(0, g)
            if score > best_score and tasks:
                best_score = score
                best_path = tasks

        return best_path

    def _generate_unsatisfiability_report(self) -> dict:
        """Generate structured explanation of why no solution path works."""
        nodes = self.db.get_all_nodes()
        pruned = [n for n in nodes if n["status"] == "pruned"]
        low_feas = [n for n in nodes if n["feasibility"] is not None and n["feasibility"] <= 2]
        errors = [n for n in nodes if n["status"] == "error"]

        report = {
            "verdict": "unsatisfiable",
            "summary": f"No viable solution path found. {len(pruned)} branches pruned, "
                       f"{len(low_feas)} nodes scored low feasibility.",
            "pruned_branches": [
                {"task": n["task"], "reason": n["pruned_reason"]}
                for n in pruned
            ],
            "low_feasibility_nodes": [
                {"task": n["task"], "score": n["feasibility"], "reason": n["feasibility_reason"]}
                for n in low_feas[:10]
            ],
            "errors": [
                {"task": n["task"], "reason": n["evaluation_reason"]}
                for n in errors
            ],
        }
        return report

    # ── Output ───────────────────────────────────────────────────────────

    def _build_output(self) -> dict:
        """Build full output with tree, solution path, and metadata."""
        def node_to_dict(node_id: int) -> dict:
            node = self.db.get_node(node_id)
            d = {
                "id": node["id"],
                "task": node["task"],
                "status": node["status"],
                "depth": node["depth"],
                "can_do_directly": bool(node["can_do_directly"]) if node["can_do_directly"] is not None else None,
                "evaluation_reason": node["evaluation_reason"],
                "relationship_to_parent": node["relationship"],
                "group": node["group_id"],
                "feasibility": node["feasibility"],
                "feasibility_reason": node["feasibility_reason"],
                "action_type": node["action_type"] or None,
                "execution_result": node["execution_result"] or None,
                "execution_success": bool(node["execution_success"]) if node["execution_success"] is not None else None,
                "mcts_visits": node["visits"],
                "mcts_value": round(node["rollout_value"], 3) if node["rollout_value"] else 0.0,
            }
            if node["pruned_reason"]:
                d["pruned_reason"] = node["pruned_reason"]
            if node["question_asked"]:
                d["question_asked"] = node["question_asked"]
            children = self.db.get_children(node_id)
            if children:
                d["children"] = [node_to_dict(c["id"]) for c in children]
            return d

        meta = self.db.get_all_metadata()

        # Extract solution path or unsatisfiability report
        solution_path = self._extract_solution_path()
        if solution_path:
            solution = {
                "verdict": "solution_found",
                "path": solution_path,
                "summary": f"Found {len(solution_path)} actionable steps.",
            }
        else:
            solution = self._generate_unsatisfiability_report()

        return {
            "metadata": meta,
            "tree": node_to_dict(0),
            "solution": solution,
        }

    def print_tree(self, output: Optional[dict] = None):
        """Pretty-print the tree and solution."""
        if output is None:
            output = self._build_output()

        tree = output.get("tree", output)

        print("\n📊 DECOMPOSITION TREE")
        print("=" * 60)
        self._print_node(tree, "", True)
        print()

        # Solution path
        solution = output.get("solution", {})
        if solution.get("verdict") == "solution_found":
            print("🎯 SOLUTION PATH")
            print("=" * 60)
            for i, step in enumerate(solution["path"], 1):
                feas = step.get("feasibility")
                feas_str = f" [{'█' * feas}{'░' * (5 - feas)}]" if feas else ""
                action = step.get("action_type", "?")
                status_icon = "✅" if step.get("status") == "executed" else "○"
                print(f"  {status_icon} {i}. [{action}]{feas_str} {step['task'][:65]}")
                if step.get("execution_result"):
                    print(f"      → {step['execution_result'][:60]}")
            print()
        else:
            print("❌ NO VIABLE SOLUTION PATH")
            print("=" * 60)
            print(f"  {solution.get('summary', 'Unknown reason')}")
            if solution.get("pruned_branches"):
                print(f"\n  Pruned branches:")
                for pb in solution["pruned_branches"][:5]:
                    print(f"    ✂ {pb['task'][:50]} — {pb['reason'][:40]}")
            print()

        # Legend
        print("Legend:")
        print("  ✅ = leaf (directly doable)    ✂  = pruned     ⚡ = executed")
        print("  🌿 = decomposed into subtasks  ⚠  = max depth / error")
        print("  ◆  = sufficient  ●  = necessary  ◐  = co-sufficient")
        print("  [█░] = feasibility (1-5)  V=visits  R=rollout value")
        print("  Action types: code | web_search | file_io | human | reasoning")
        print()

    def _print_node(self, node: dict, prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "

        icon = {
            "leaf": "✅", "decomposed": "🌿", "max-depth": "⚠ ",
            "error": "❌", "pending": "⏳", "pruned": "✂ ", "executed": "⚡",
        }.get(node["status"], "?")

        rel = node.get("relationship_to_parent", "")
        group = node.get("group", 0)
        tag = ""
        if rel:
            symbol = {"sufficient": "◆", "necessary": "●", "co-sufficient": "◐"}.get(rel, "?")
            tag = f" {symbol}[{rel}|g{group}]"

        feas = node.get("feasibility")
        feas_tag = f" [{'█' * feas}{'░' * (5 - feas)}]" if feas else ""

        action_tag = f" [{node['action_type']}]" if node.get("action_type") else ""

        visits = node.get("mcts_visits", 0)
        value = node.get("mcts_value", 0.0)
        mcts_tag = f" V={visits}" if visits else ""

        task_display = node["task"][:50] + "..." if len(node["task"]) > 50 else node["task"]

        if node["depth"] == 0:
            print(f"{icon} {task_display}")
        else:
            print(f"{prefix}{connector}{icon}{tag}{feas_tag}{action_tag}{mcts_tag} {task_display}")

        if node.get("pruned_reason"):
            child_prefix = prefix + ("    " if is_last else "│   ")
            print(f"{child_prefix}  ↳ {node['pruned_reason'][:60]}")

        children = node.get("children", [])
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            if node["depth"] == 0:
                child_prefix = prefix
            else:
                child_prefix = prefix + ("    " if is_last else "│   ")
            self._print_node(child, child_prefix, child_is_last)


class BudgetExhausted(Exception):
    """Raised when any resource budget is exhausted."""
    pass


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BBQ Stage 3 — MCTS Decomposer with Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python bbq3.py "Make me $1000 legally by today"\n'
            '  python bbq3.py --budget-time 300 --mcts-iter 20 "Your complex task"\n'
            '  python bbq3.py --no-execute -o tree.json "Plan only, no execution"\n'
            '  python bbq3.py -v --db tree.db --viewer "Full run with viewer"'
        ),
    )
    parser.add_argument("task", help="The complex task to decompose")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                        help=f"Max tree depth (default: {MAX_DEPTH})")
    parser.add_argument("--max-children", type=int, default=MAX_CHILDREN,
                        help=f"Max children per node (default: {MAX_CHILDREN})")
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES,
                        help=f"Max total nodes (default: {MAX_NODES})")
    parser.add_argument("--min-feasibility", type=int, default=MIN_FEASIBILITY,
                        choices=[1, 2, 3, 4, 5],
                        help=f"Prune nodes below this feasibility (default: {MIN_FEASIBILITY})")
    parser.add_argument("--mcts-iter", type=int, default=MCTS_ITERATIONS,
                        help=f"MCTS iterations (default: {MCTS_ITERATIONS})")
    parser.add_argument("--budget-time", type=int, default=BUDGET_TIME,
                        help=f"Max wall-clock seconds (default: {BUDGET_TIME})")
    parser.add_argument("--budget-calls", type=int, default=BUDGET_CALLS,
                        help=f"Max LLM API calls (default: {BUDGET_CALLS})")
    parser.add_argument("--no-execute", action="store_true",
                        help="Disable execution — plan only")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Anthropic model (default: {DEFAULT_MODEL})")
    parser.add_argument("--db", default=None,
                        help="Save SQLite database to file")
    parser.add_argument("-o", "--output", help="Save JSON output to file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show LLM call details and MCTS internals")
    parser.add_argument("--viewer", action="store_true",
                        help="Open interactive viewer in browser after decomposition")

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    db_path = args.db if args.db else ":memory:"

    decomposer = BBQDecomposer3(
        model=args.model,
        max_depth=args.max_depth,
        max_children=args.max_children,
        max_nodes=args.max_nodes,
        min_feasibility=args.min_feasibility,
        mcts_iterations=args.mcts_iter,
        budget_time=args.budget_time,
        budget_calls=args.budget_calls,
        execute=not args.no_execute,
        verbose=args.verbose,
        db_path=db_path,
    )

    result = decomposer.decompose(args.task)
    decomposer.print_tree(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 Tree saved to {args.output}")

    if args.db:
        print(f"💾 Database saved to {args.db}")

    if args.viewer:
        import webbrowser
        import pathlib
        viewer_json = pathlib.Path(__file__).parent / ".bbq_last_output.json"
        with open(viewer_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        viewer_html = pathlib.Path(__file__).parent / "viewer.html"
        if viewer_html.exists():
            webbrowser.open(viewer_html.as_uri() + "?file=.bbq_last_output.json")
            print(f"🌐 Opened viewer in browser")
        else:
            print(f"⚠ viewer.html not found at {viewer_html}")

    decomposer.db.close()


if __name__ == "__main__":
    main()
