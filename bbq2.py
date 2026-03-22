#!/usr/bin/env python3
"""
BBQ Stage 2 — Back Branching Questioning Decomposer

Builds on Stage 1 with:
  - Feasibility scoring (1-5) per node via a 4th LLM call
  - Pruning: rule violations, impossibilities, low-feasibility cutoff
  - SQLite persistence (nodes + edges tables)
  - Enhanced JSON export with full metadata

Usage:
  python bbq2.py "Make me $1000 legally by today"
  python bbq2.py --max-depth 3 --min-feasibility 2 "Design a mobile app"
  python bbq2.py -v --db tree.db -o tree.json "Your complex task"
"""

import argparse
import json
import sqlite3
import sys
import os
import time

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from typing import Optional
from anthropic import Anthropic

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_DEPTH = 4
MAX_CHILDREN = 5
MAX_NODES = 50
MIN_FEASIBILITY = 2  # prune nodes scoring below this

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
                id              INTEGER PRIMARY KEY,
                task            TEXT NOT NULL,
                depth           INTEGER NOT NULL DEFAULT 0,
                status          TEXT NOT NULL DEFAULT 'pending',
                can_do_directly INTEGER,
                evaluation_reason TEXT DEFAULT '',
                question_asked  TEXT DEFAULT '',
                relationship    TEXT DEFAULT '',
                group_id        INTEGER DEFAULT 0,
                feasibility     INTEGER,
                feasibility_reason TEXT DEFAULT '',
                pruned_reason   TEXT DEFAULT '',
                created_at      REAL NOT NULL
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


# ── Decomposer ──────────────────────────────────────────────────────────────

class BBQDecomposer2:
    def __init__(self, model: str = DEFAULT_MODEL, max_depth: int = MAX_DEPTH,
                 max_children: int = MAX_CHILDREN, max_nodes: int = MAX_NODES,
                 min_feasibility: int = MIN_FEASIBILITY,
                 verbose: bool = False, db_path: str = ":memory:"):
        self.client = Anthropic()
        self.model = model
        self.max_depth = max_depth
        self.max_children = max_children
        self.max_nodes = max_nodes
        self.min_feasibility = min_feasibility
        self.verbose = verbose
        self.db = TreeDB(db_path)
        self.next_id = 0
        self.total_calls = 0
        self.total_pruned = 0

    # ── LLM interface ────────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Single LLM call with minimal context."""
        self.total_calls += 1
        if self.verbose:
            print(f"  [LLM call #{self.total_calls}]")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.3,
        )
        return response.content[0].text

    # ── The four calls ────────────────────────────────────────────────────

    def _evaluate(self, node_id: int, path_context: str) -> tuple[bool, str]:
        """Call 1: Can this task be done directly by an LLM agent?"""
        node = self.db.get_node(node_id)
        system = (
            "You are an evaluator for a task decomposition system. "
            "You assess whether a task can be accomplished DIRECTLY in a single step "
            "by an LLM agent (with access to common tools like web search, code execution, file I/O). "
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
        """Call 2: What's the best question to find causal precursors?"""
        node = self.db.get_node(node_id)
        system = (
            "You are a strategic question designer for a task decomposition system called BBQ "
            "(Back Branching Questioning). Your job is to find the SINGLE BEST question whose "
            "answer would reveal all the causal precursors — the things that need to be true "
            "or need to happen — for a given task to be achieved.\n\n"
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
            "Group co-sufficient and necessary tasks into GROUPS. Each group represents "
            "one complete path to achieving the parent. A group might be:\n"
            "- A single sufficient task (group by itself)\n"
            "- A set of necessary + co-sufficient tasks that together are sufficient\n\n"
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

        # Parse JSON — handle possible markdown fences
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
        """Call 4 (NEW): Rate feasibility 1-5 for this subtask."""
        node = self.db.get_node(node_id)
        system = (
            "You are a feasibility assessor for a task decomposition system. "
            "Rate how feasible the given task is on a 1-5 scale:\n\n"
            "5 = Straightforward — well-understood, can be done with standard methods\n"
            "4 = Doable with effort — requires skill or resources but clearly achievable\n"
            "3 = Uncertain — requires significant research, resources, or luck\n"
            "2 = Unlikely — major obstacles, would require breakthroughs or extraordinary circumstances\n"
            "1 = Near-impossible — technology doesn't exist, violates known constraints, or is practically unachievable\n\n"
            "Consider:\n"
            "- Does the required technology/knowledge exist today?\n"
            "- Are the resources (time, money, access) realistic?\n"
            "- Does this conflict with any constraints stated in the root task?\n"
            "- How many things need to go right for this to work?\n\n"
            "Respond in exactly this format:\n"
            "SCORE: <number 1-5>\n"
            "REASON: one sentence explaining the rating"
        )
        user = f"Task context (path from root):\n{path_context}\n\nTask to assess:\n{node['task']}"

        result = self._call_llm(system, user)

        # Parse score
        score = 3  # default if parsing fails
        for line in result.splitlines():
            if line.strip().upper().startswith("SCORE"):
                for ch in line:
                    if ch.isdigit() and 1 <= int(ch) <= 5:
                        score = int(ch)
                        break
                break
        reason = result.split("REASON:")[-1].strip() if "REASON:" in result else result
        return score, reason

    def _check_pruning(self, node_id: int, root_task: str, path_context: str) -> tuple[bool, str]:
        """Check if a node should be pruned for rule violations or contradictions."""
        node = self.db.get_node(node_id)

        # First check: feasibility cutoff
        if node["feasibility"] is not None and node["feasibility"] < self.min_feasibility:
            return True, f"feasibility too low ({node['feasibility']}/{self.min_feasibility} min)"

        # Second check: LLM-based rule violation detection
        system = (
            "You are a constraint checker for a task decomposition system. "
            "You check whether a subtask CONTRADICTS or VIOLATES any constraints "
            "implied by the root task.\n\n"
            "Examples of violations:\n"
            "- Root says 'legally' but subtask involves illegal activity\n"
            "- Root says 'by today' but subtask requires months of work\n"
            "- Subtask is logically impossible or self-contradictory\n\n"
            "Respond in exactly this format:\n"
            "VERDICT: VALID or VIOLATION\n"
            "REASON: one sentence explaining why"
        )
        user = (
            f"Root task:\n{root_task}\n\n"
            f"Task context (path from root):\n{path_context}\n\n"
            f"Subtask to check:\n{node['task']}"
        )

        result = self._call_llm(system, user)

        is_violation = "VIOLATION" in result.split("REASON")[0].upper()
        reason = result.split("REASON:")[-1].strip() if "REASON:" in result else result
        if is_violation:
            return True, f"rule violation: {reason}"
        return False, ""

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

    # ── Main loop ────────────────────────────────────────────────────────

    def decompose(self, task: str) -> dict:
        """Run the full BBQ Stage 2 decomposition."""
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"BBQ Decomposer — Stage 2")
        print(f"{'='*60}")
        print(f"Task: {task}")
        print(f"Limits: depth={self.max_depth}, children={self.max_children}, nodes={self.max_nodes}")
        print(f"Feasibility cutoff: {self.min_feasibility}/5")
        print(f"{'='*60}\n")

        # Store root task for pruning checks
        self.root_task = task

        # Save metadata
        self.db.set_metadata("root_task", task)
        self.db.set_metadata("model", self.model)
        self.db.set_metadata("max_depth", str(self.max_depth))
        self.db.set_metadata("max_children", str(self.max_children))
        self.db.set_metadata("max_nodes", str(self.max_nodes))
        self.db.set_metadata("min_feasibility", str(self.min_feasibility))
        self.db.set_metadata("start_time", str(start_time))

        # Create root
        root_id = self._create_node(task)

        # BFS queue
        queue = [root_id]

        while queue and self.db.count_nodes() < self.max_nodes:
            node_id = queue.pop(0)
            node = self.db.get_node(node_id)

            if node["status"] == "pruned":
                continue

            if node["depth"] >= self.max_depth:
                self.db.update_node(node_id, status="max-depth")
                if self.verbose:
                    print(f"  ⚠ Max depth reached for: {node['task'][:50]}...")
                continue

            path_context = self._get_path_context(node_id)
            indent = "  " * node["depth"]

            # ── Call 1: Evaluate ──
            print(f"{indent}📋 Evaluating: {node['task'][:60]}{'...' if len(node['task']) > 60 else ''}")
            can_do, reason = self._evaluate(node_id, path_context)
            self.db.update_node(node_id,
                                can_do_directly=1 if can_do else 0,
                                evaluation_reason=reason)

            if can_do:
                self.db.update_node(node_id, status="leaf")
                print(f"{indent}   ✅ Leaf — {reason[:60]}")
                continue

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
                continue

            self.db.update_node(node_id, status="decomposed")

            for st in subtasks:
                if self.db.count_nodes() >= self.max_nodes:
                    print(f"\n⚠ Node budget exhausted ({self.max_nodes} nodes)")
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
                self.db.update_node(child_id,
                                    feasibility=score,
                                    feasibility_reason=score_reason)

                score_bar = "█" * score + "░" * (5 - score)
                rel_symbol = {"sufficient": "◆", "necessary": "●", "co-sufficient": "◐"}.get(st["relationship"], "?")
                print(f"{indent}   {rel_symbol} [{st['relationship']}|g{st['group']}] [{score_bar}] {st['description'][:45]}")

                # ── Pruning check ──
                should_prune, prune_reason = self._check_pruning(child_id, self.root_task, child_path)
                if should_prune:
                    self.db.update_node(child_id, status="pruned", pruned_reason=prune_reason)
                    self.total_pruned += 1
                    print(f"{indent}      ✂ PRUNED — {prune_reason[:55]}")
                else:
                    queue.append(child_id)

            print()

        elapsed = time.time() - start_time
        self.db.set_metadata("end_time", str(time.time()))
        self.db.set_metadata("total_calls", str(self.total_calls))
        self.db.set_metadata("total_nodes", str(self.db.count_nodes()))
        self.db.set_metadata("total_pruned", str(self.total_pruned))
        self.db.set_metadata("elapsed_seconds", f"{elapsed:.1f}")

        print(f"\n{'='*60}")
        print(f"Done. {self.db.count_nodes()} nodes, {self.total_calls} LLM calls, "
              f"{self.total_pruned} pruned, {elapsed:.1f}s")
        print(f"{'='*60}\n")

        return self._build_output()

    # ── Output ───────────────────────────────────────────────────────────

    def _build_output(self) -> dict:
        """Build the full tree as a nested dict for JSON export."""
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
        return {
            "metadata": meta,
            "tree": node_to_dict(0),
        }

    def print_tree(self, output: Optional[dict] = None):
        """Pretty-print the decomposition tree."""
        if output is None:
            output = self._build_output()

        tree = output.get("tree", output)

        print("\n📊 DECOMPOSITION TREE")
        print("=" * 60)
        self._print_node(tree, "", True)
        print()

        # Print legend
        print("Legend:")
        print("  ✅ = leaf (directly doable)    ✂  = pruned")
        print("  🌿 = decomposed into subtasks  ⚠  = max depth / error")
        print("  ◆  = sufficient  ●  = necessary  ◐  = co-sufficient")
        print("  [█░] = feasibility score (1-5)")
        print("  g1, g2... = sufficiency groups")
        print()

    def _print_node(self, node: dict, prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "

        # Status icon
        icon = {
            "leaf": "✅",
            "decomposed": "🌿",
            "max-depth": "⚠ ",
            "error": "❌",
            "pending": "⏳",
            "pruned": "✂ ",
        }.get(node["status"], "?")

        # Relationship tag
        rel = node.get("relationship_to_parent", "")
        group = node.get("group", 0)
        tag = ""
        if rel:
            symbol = {"sufficient": "◆", "necessary": "●", "co-sufficient": "◐"}.get(rel, "?")
            tag = f" {symbol}[{rel}|g{group}]"

        # Feasibility bar
        feas = node.get("feasibility")
        feas_tag = ""
        if feas is not None:
            feas_tag = f" [{'█' * feas}{'░' * (5 - feas)}]"

        # Task display
        task_display = node["task"][:55]
        if len(node["task"]) > 55:
            task_display += "..."

        if node["depth"] == 0:
            print(f"{icon} {task_display}")
        else:
            print(f"{prefix}{connector}{icon}{tag}{feas_tag} {task_display}")

        # Pruning reason
        if node.get("pruned_reason"):
            child_prefix = prefix + ("    " if is_last else "│   ")
            print(f"{child_prefix}  ↳ {node['pruned_reason'][:60]}")

        # Print children
        children = node.get("children", [])
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            if node["depth"] == 0:
                child_prefix = prefix
            else:
                child_prefix = prefix + ("    " if is_last else "│   ")
            self._print_node(child, child_prefix, child_is_last)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BBQ Stage 2 — Back Branching Questioning Decomposer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python bbq2.py "Make me $1000 legally by today"\n'
            '  python bbq2.py --max-depth 3 --min-feasibility 2 "Design a mobile app"\n'
            '  python bbq2.py -v --db tree.db -o tree.json "Your complex task"'
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
                        help=f"Prune nodes below this feasibility score (default: {MIN_FEASIBILITY})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Anthropic model (default: {DEFAULT_MODEL})")
    parser.add_argument("--db", default=None,
                        help="Save SQLite database to file (default: in-memory only)")
    parser.add_argument("-o", "--output", help="Save JSON output to file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show LLM call details")

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    db_path = args.db if args.db else ":memory:"

    decomposer = BBQDecomposer2(
        model=args.model,
        max_depth=args.max_depth,
        max_children=args.max_children,
        max_nodes=args.max_nodes,
        min_feasibility=args.min_feasibility,
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

    decomposer.db.close()


if __name__ == "__main__":
    main()
