#!/usr/bin/env python3
"""
BBQ Stage 1 — Back Branching Questioning Decomposer

Takes a complex task and decomposes it into a tree of subtasks
using three focused LLM calls per node:
  1. Evaluator  — "Can this be done directly?"
  2. Question finder — "What question reveals the precursors?"
  3. Decomposer — "Answer the question → subtasks"

Usage:
  python bbq.py "Make me $1000 legally by today"
  python bbq.py --max-depth 3 --max-children 4 "Design a mobile app for elderly care"
"""

import argparse
import json
import sys
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from anthropic import Anthropic

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_DEPTH = 4
MAX_CHILDREN = 5
MAX_NODES = 50

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SubTask:
    description: str
    relationship: str  # "sufficient", "necessary", "co-sufficient"
    group: int  # which sufficiency group this belongs to

@dataclass
class Node:
    id: int
    task: str
    parent_id: Optional[int] = None
    depth: int = 0
    status: str = "pending"  # pending, leaf, decomposed, pruned
    can_do_directly: Optional[bool] = None
    evaluation_reason: str = ""
    question_asked: str = ""
    children: list = field(default_factory=list)  # list of child node ids
    relationship_to_parent: str = ""  # how achieving this helps the parent
    group: int = 0  # sufficiency group within parent


class BBQDecomposer:
    def __init__(self, model: str = DEFAULT_MODEL, max_depth: int = MAX_DEPTH,
                 max_children: int = MAX_CHILDREN, max_nodes: int = MAX_NODES,
                 verbose: bool = False):
        self.client = Anthropic()
        self.model = model
        self.max_depth = max_depth
        self.max_children = max_children
        self.max_nodes = max_nodes
        self.verbose = verbose
        self.nodes: dict[int, Node] = {}
        self.next_id = 0
        self.total_calls = 0

    # ── LLM interface ────────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str) -> str:
        """Single LLM call with minimal context."""
        self.total_calls += 1
        if self.verbose:
            print(f"  [LLM call #{self.total_calls}]")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.3,  # low temp for structured reasoning
        )
        return response.content[0].text

    # ── The three calls ──────────────────────────────────────────────────

    def _evaluate(self, node: Node, path_context: str) -> tuple[bool, str]:
        """Call 1: Can this task be done directly by an LLM agent?"""
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
        user = f"Task context (path from root):\n{path_context}\n\nTask to evaluate:\n{node.task}"
        
        result = self._call_llm(system, user)
        
        verdict = "YES" in result.split("REASON")[0].upper()
        reason = result.split("REASON:")[-1].strip() if "REASON:" in result else result
        return verdict, reason

    def _find_question(self, node: Node, path_context: str) -> str:
        """Call 2: What's the best question to find causal precursors?"""
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
        user = f"Task context (path from root):\n{path_context}\n\nTask to decompose:\n{node.task}"
        
        return self._call_llm(system, user).strip().strip('"')

    def _decompose(self, node: Node, question: str, path_context: str) -> list[SubTask]:
        """Call 3: Answer the question to produce classified subtasks."""
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
            f"Task to decompose:\n{node.task}\n\n"
            f"Guiding question:\n{question}"
        )
        
        result = self._call_llm(system, user)
        
        # Parse JSON — handle possible markdown fences
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
        
        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON array
            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1
            if start >= 0 and end > start:
                items = json.loads(cleaned[start:end])
            else:
                raise ValueError(f"Could not parse decomposer output:\n{result}")
        
        return [
            SubTask(
                description=item["description"],
                relationship=item.get("relationship", "sufficient"),
                group=item.get("group", 1),
            )
            for item in items[:self.max_children]
        ]

    # ── Tree operations ──────────────────────────────────────────────────

    def _create_node(self, task: str, parent_id: Optional[int] = None,
                     relationship: str = "", group: int = 0) -> Node:
        node = Node(
            id=self.next_id,
            task=task,
            parent_id=parent_id,
            depth=0 if parent_id is None else self.nodes[parent_id].depth + 1,
            relationship_to_parent=relationship,
            group=group,
        )
        self.next_id += 1
        self.nodes[node.id] = node
        if parent_id is not None:
            self.nodes[parent_id].children.append(node.id)
        return node

    def _get_path_context(self, node: Node) -> str:
        """Build minimal context: just the task descriptions from root to this node."""
        path = []
        current = node
        while current is not None:
            path.append(current.task)
            current = self.nodes.get(current.parent_id) if current.parent_id is not None else None
        path.reverse()
        return " → ".join(path)

    # ── Main loop ────────────────────────────────────────────────────────

    def decompose(self, task: str) -> dict:
        """Run the full BBQ decomposition. Returns the tree as a dict."""
        
        print(f"\n{'='*60}")
        print(f"BBQ Decomposer — Stage 1")
        print(f"{'='*60}")
        print(f"Task: {task}")
        print(f"Limits: depth={self.max_depth}, children={self.max_children}, nodes={self.max_nodes}")
        print(f"{'='*60}\n")

        # Create root
        root = self._create_node(task)
        
        # BFS queue
        queue = [root.id]
        
        while queue and len(self.nodes) < self.max_nodes:
            node_id = queue.pop(0)
            node = self.nodes[node_id]
            
            if node.depth >= self.max_depth:
                node.status = "max-depth"
                if self.verbose:
                    print(f"  ⚠ Max depth reached for: {node.task[:50]}...")
                continue
            
            path_context = self._get_path_context(node)
            indent = "  " * node.depth
            
            # ── Call 1: Evaluate ──
            print(f"{indent}📋 Evaluating: {node.task[:60]}{'...' if len(node.task) > 60 else ''}")
            can_do, reason = self._evaluate(node, path_context)
            node.can_do_directly = can_do
            node.evaluation_reason = reason
            
            if can_do:
                node.status = "leaf"
                print(f"{indent}   ✅ Leaf — {reason[:60]}")
                continue
            
            print(f"{indent}   ❌ Cannot do directly — {reason[:60]}")
            
            # ── Call 2: Find question ──
            print(f"{indent}   🔍 Finding decomposition question...")
            question = self._find_question(node, path_context)
            node.question_asked = question
            print(f"{indent}   ❓ \"{question[:70]}{'...' if len(question) > 70 else ''}\"")
            
            # ── Call 3: Decompose ──
            print(f"{indent}   🌿 Decomposing...")
            try:
                subtasks = self._decompose(node, question, path_context)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"{indent}   ⚠ Decomposition failed: {e}")
                node.status = "error"
                continue
            
            node.status = "decomposed"
            
            for st in subtasks:
                if len(self.nodes) >= self.max_nodes:
                    print(f"\n⚠ Node budget exhausted ({self.max_nodes} nodes)")
                    break
                child = self._create_node(
                    task=st.description,
                    parent_id=node.id,
                    relationship=st.relationship,
                    group=st.group,
                )
                queue.append(child.id)
                rel_symbol = {"sufficient": "◆", "necessary": "●", "co-sufficient": "◐"}.get(st.relationship, "?")
                print(f"{indent}   {rel_symbol} [{st.relationship}|g{st.group}] {st.description[:55]}")
            
            print()
        
        print(f"\n{'='*60}")
        print(f"Done. {len(self.nodes)} nodes, {self.total_calls} LLM calls")
        print(f"{'='*60}\n")
        
        return self._build_output()

    # ── Output ───────────────────────────────────────────────────────────

    def _build_output(self) -> dict:
        """Build the full tree as a nested dict for JSON export."""
        def node_to_dict(node_id: int) -> dict:
            node = self.nodes[node_id]
            d = {
                "id": node.id,
                "task": node.task,
                "status": node.status,
                "depth": node.depth,
                "can_do_directly": node.can_do_directly,
                "evaluation_reason": node.evaluation_reason,
                "relationship_to_parent": node.relationship_to_parent,
                "group": node.group,
            }
            if node.question_asked:
                d["question_asked"] = node.question_asked
            if node.children:
                d["children"] = [node_to_dict(cid) for cid in node.children]
            return d
        
        return node_to_dict(0)

    def print_tree(self, output: Optional[dict] = None):
        """Pretty-print the decomposition tree."""
        if output is None:
            output = self._build_output()
        
        print("\n📊 DECOMPOSITION TREE")
        print("=" * 60)
        self._print_node(output, "", True)
        print()
        
        # Print legend
        print("Legend:")
        print("  ✅ = leaf (directly doable)")
        print("  🌿 = decomposed into subtasks")
        print("  ⚠  = max depth / error")
        print("  ◆  = sufficient  ●  = necessary  ◐  = co-sufficient")
        print("  g1, g2... = sufficiency groups (any one group achieves the parent)")
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
        }.get(node["status"], "?")
        
        # Relationship tag
        rel = node.get("relationship_to_parent", "")
        group = node.get("group", 0)
        tag = ""
        if rel:
            symbol = {"sufficient": "◆", "necessary": "●", "co-sufficient": "◐"}.get(rel, "?")
            tag = f" {symbol}[{rel}|g{group}]"
        
        # Print node
        task_display = node["task"][:65]
        if len(node["task"]) > 65:
            task_display += "..."
        
        if node["depth"] == 0:
            print(f"{icon} {task_display}")
        else:
            print(f"{prefix}{connector}{icon}{tag} {task_display}")
        
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
        description="BBQ Stage 1 — Back Branching Questioning Decomposer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python bbq.py "Make me $1000 legally by today"\n'
            '  python bbq.py --max-depth 3 "Design a mobile app for elderly care"\n'
            '  python bbq.py -v -o tree.json "Plan a wedding for 200 guests in 2 months"'
        ),
    )
    parser.add_argument("task", help="The complex task to decompose")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH, help=f"Max tree depth (default: {MAX_DEPTH})")
    parser.add_argument("--max-children", type=int, default=MAX_CHILDREN, help=f"Max children per node (default: {MAX_CHILDREN})")
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES, help=f"Max total nodes (default: {MAX_NODES})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Anthropic model (default: {DEFAULT_MODEL})")
    parser.add_argument("-o", "--output", help="Save JSON output to file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show LLM call details")
    
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    
    decomposer = BBQDecomposer(
        model=args.model,
        max_depth=args.max_depth,
        max_children=args.max_children,
        max_nodes=args.max_nodes,
        verbose=args.verbose,
    )
    
    result = decomposer.decompose(args.task)
    decomposer.print_tree(result)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"💾 Tree saved to {args.output}")


if __name__ == "__main__":
    main()
