from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .models import EdgeInfo


PathTuple = Tuple[List[str], float, List[EdgeInfo]]


def simple_paths_topk(
    graph: Dict[str, List[EdgeInfo]],
    start: str,
    max_hops: int,
    k: int,
) -> List[PathTuple]:
    results: List[PathTuple] = []
    q = deque([(start, [start], [])])

    while q:
        node, path, edges = q.popleft()
        if len(path) > 1:
            # Path score is the average Jaccard of the joins along the path.
            score = float(sum(e.score for e in edges) / len(edges))
            results.append((path, score, edges))

        if len(path) - 1 >= max_hops:
            continue

        for edge in graph.get(node, []):
            if edge.left == node:
                nxt = edge.right
            elif edge.right == node:
                nxt = edge.left
            else:
                continue
            if nxt in path:
                continue
            q.append((nxt, path + [nxt], edges + [edge]))

    by_hops: Dict[int, List[PathTuple]] = defaultdict(list)
    for item in results:
        by_hops[len(item[0]) - 1].append(item)
    for hop_count in by_hops:
        by_hops[hop_count].sort(key=lambda x: x[1], reverse=True)

    picked: List[PathTuple] = []
    seen: Set[Tuple[str, ...]] = set()
    for hop_count in range(1, max_hops + 1):
        items = by_hops.get(hop_count, [])
        if not items:
            continue
        item = items[0]
        key = tuple(item[0])
        if key not in seen:
            picked.append(item)
            seen.add(key)
        if len(picked) >= k:
            return picked[:k]

    leftovers = [item for items in by_hops.values() for item in items]
    leftovers.sort(key=lambda x: x[1], reverse=True)
    for item in leftovers:
        if len(picked) >= k:
            break
        key = tuple(item[0])
        if key in seen:
            continue
        picked.append(item)
        seen.add(key)

    return picked[:k]


def diversify_paths_by_node_count(
    paths: Sequence[PathTuple],
    *,
    min_nodes: int,
    max_nodes: int,
    k: int,
) -> List[PathTuple]:
    by_n: Dict[int, List[PathTuple]] = defaultdict(list)
    for nodes, score, edges in paths:
        if min_nodes <= len(nodes) <= max_nodes:
            by_n[len(nodes)].append((nodes, score, edges))
    for n in by_n:
        by_n[n].sort(key=lambda x: x[1], reverse=True)

    picked: List[PathTuple] = []
    seen: Set[Tuple[str, ...]] = set()
    idx_by_n = {n: 0 for n in range(min_nodes, max_nodes + 1)}

    while len(picked) < k:
        progressed = False
        for n in range(min_nodes, max_nodes + 1):
            items = by_n.get(n, [])
            i = idx_by_n.get(n, 0)
            while i < len(items) and tuple(items[i][0]) in seen:
                i += 1
            idx_by_n[n] = i
            if i >= len(items):
                continue
            item = items[i]
            idx_by_n[n] = i + 1
            picked.append(item)
            seen.add(tuple(item[0]))
            progressed = True
            if len(picked) >= k:
                break
        if not progressed:
            break
    return picked[:k]


def materialize_path(
    tables: Dict[str, pd.DataFrame],
    path_nodes: Sequence[str],
    path_edges: Sequence[EdgeInfo],
) -> pd.DataFrame:
    if not path_nodes:
        return pd.DataFrame()

    df = tables[path_nodes[0]].copy()
    for edge, next_node in zip(path_edges, path_nodes[1:]):
        right = tables[next_node]
        left_on = str(edge.key)
        right_on = str(edge.right_key or edge.key)

        if left_on not in df.columns:
            df[left_on] = right[right_on].iloc[0] if right_on in right.columns and len(right) else ""
        if right_on not in right.columns and left_on in right.columns:
            right_on = left_on

        df = df.merge(
            right,
            how="left",
            left_on=left_on,
            right_on=right_on,
            suffixes=("", f"_{next_node}"),
        )
    return df


def format_path(path: Sequence[str], edges: Sequence[EdgeInfo]) -> str:
    parts = [path[0]]
    for edge, nxt in zip(edges, path[1:]):
        parts.append(f"--[{edge.join_type}:{edge.key}->{edge.right_key or edge.key}, {edge.score:.2f}]-- {nxt}")
    return " ".join(parts)


def path_to_dot(nodes: Sequence[str], edges: Sequence[EdgeInfo], tables: Optional[Dict[str, pd.DataFrame]] = None) -> str:
    lines = [
        "graph P {",
        '  graph [rankdir="LR", bgcolor="white", ranksep="0.12", nodesep="0.10"];',
        '  node [shape="box", style="rounded,filled", fillcolor="#f8fafc", color="#475569", fontname="Helvetica", fontsize="9"];',
        '  edge [fontname="Helvetica", fontsize="8"];',
    ]
    for node in nodes:
        lines.append(f'  "{node}";')
    for i, (edge, nxt) in enumerate(zip(edges, nodes[1:])):
        label = f"{edge.key}, {edge.right_key or edge.key}"
        color = "#2563EB" if edge.join_type == "equi" else "#F97316"
        lines.append(f'  "{nodes[i]}" -- "{nxt}" [label="{label}", color="{color}"];')
    lines.append("}")
    return "\n".join(lines)


def path_to_dot_highlighted(
    nodes: Sequence[str],
    edges: Sequence[EdgeInfo],
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    highlight_blue: bool = False,
    highlight_green_nodes: Optional[Set[str]] = None,
) -> str:
    return path_to_dot(nodes, edges, tables)

