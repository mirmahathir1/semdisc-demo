from __future__ import annotations

import io
import itertools
import hashlib
import time
import zipfile
from contextlib import nullcontext
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class EdgeInfo:
    left: str
    right: str
    key: str
    score: float


def normalize_cell(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = {x for x in a if x}
    sb = {x for x in b if x}
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def best_join_key_and_score(left: pd.DataFrame, right: pd.DataFrame) -> Optional[Tuple[str, float]]:
    common_cols = [c for c in left.columns if c in right.columns]
    if not common_cols:
        return None

    best = None
    for col in common_cols:
        lvals = (normalize_cell(v) for v in left[col].head(5000))
        rvals = (normalize_cell(v) for v in right[col].head(5000))
        score = jaccard(lvals, rvals)
        if best is None or score > best[1]:
            best = (col, score)
    return best


def build_join_graph(tables: Dict[str, pd.DataFrame], min_edge_score: float) -> Dict[str, List[EdgeInfo]]:
    graph: Dict[str, List[EdgeInfo]] = defaultdict(list)
    names = sorted(tables.keys())
    for a, b in itertools.combinations(names, 2):
        best = best_join_key_and_score(tables[a], tables[b])
        if best is None:
            continue
        key, score = best
        if score < min_edge_score:
            continue
        graph[a].append(EdgeInfo(left=a, right=b, key=key, score=score))
        graph[b].append(EdgeInfo(left=b, right=a, key=key, score=score))
    return graph


def attach_query_node(
    graph: Dict[str, List[EdgeInfo]],
    tables: Dict[str, pd.DataFrame],
    query: pd.DataFrame,
    min_edge_score: float,
) -> Tuple[Dict[str, List[EdgeInfo]], Dict[str, pd.DataFrame]]:
    tables2 = dict(tables)
    tables2["query"] = query
    g2: Dict[str, List[EdgeInfo]] = defaultdict(list, {k: list(v) for k, v in graph.items()})

    for name, df in tables.items():
        best = best_join_key_and_score(query, df)
        if best is None:
            continue
        key, score = best
        if score < min_edge_score:
            continue
        g2["query"].append(EdgeInfo(left="query", right=name, key=key, score=score))
        g2[name].append(EdgeInfo(left=name, right="query", key=key, score=score))

    return g2, tables2


def simple_paths_topk(
    graph: Dict[str, List[EdgeInfo]],
    start: str,
    max_hops: int,
    k: int,
) -> List[Tuple[List[str], float, List[EdgeInfo]]]:
    # Rank by multiplicative score with a mild preference for shorter paths.
    results: List[Tuple[List[str], float, List[EdgeInfo]]] = []
    q = deque()
    q.append((start, [start], 1.0, []))  # node, path, score, edges

    while q:
        node, path, score, edges = q.popleft()
        if len(path) > 1:
            hop_count = len(path) - 1
            adjusted = score * (1.0 / (1.0 + 0.15 * hop_count))
            results.append((path, adjusted, edges))

        if len(path) - 1 >= max_hops:
            continue

        for e in graph.get(node, []):
            nxt = e.right
            if nxt in path:
                continue
            q.append((nxt, path + [nxt], score * max(e.score, 1e-9), edges + [e]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


def materialize_path(
    tables: Dict[str, pd.DataFrame],
    path_nodes: Sequence[str],
    path_edges: Sequence[EdgeInfo],
) -> pd.DataFrame:
    df = tables[path_nodes[0]].copy()
    for edge, next_node in zip(path_edges, path_nodes[1:]):
        right = tables[next_node]
        key = edge.key
        df = df.merge(right, how="left", on=key, suffixes=("", f"_{next_node}"))
    return df


def to_dot(
    graph: Dict[str, List[EdgeInfo]],
    highlight_edges: Sequence[Tuple[str, str]] = (),
) -> str:
    highlight = set()
    for a, b in highlight_edges:
        highlight.add((a, b))
        highlight.add((b, a))

    # Build a stable edge set to avoid double-printing undirected edges.
    undirected = {}
    for a, edges in graph.items():
        for e in edges:
            key = tuple(sorted([e.left, e.right]))
            undirected[key] = e

    lines = [
        "graph G {",
        '  graph [rankdir="LR", bgcolor="white"];',
        '  node [shape="box", style="rounded,filled", fillcolor="#f7f7f7", color="#444444", fontname="Helvetica"];',
        '  edge [color="#666666", fontname="Helvetica"];',
    ]

    for (u, v), e in sorted(undirected.items(), key=lambda kv: kv[0]):
        is_hl = (u, v) in highlight or (v, u) in highlight
        color = "#d62728" if is_hl else "#666666"
        penwidth = "3.0" if is_hl else "1.2"
        label = f"{e.key}\\n{e.score:.2f}"
        lines.append(f'  "{u}" -- "{v}" [label="{label}", color="{color}", penwidth="{penwidth}"];')

    lines.append("}")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def bundled_datalake() -> Dict[str, pd.DataFrame]:
    customers = pd.DataFrame(
        [
            {"customer_id": "C001", "name": "A. Rivera", "country": "US"},
            {"customer_id": "C002", "name": "S. Tan", "country": "SG"},
            {"customer_id": "C003", "name": "M. Kowalski", "country": "PL"},
        ]
    )
    orders = pd.DataFrame(
        [
            {"order_id": "O100", "customer_id": "C001", "order_date": "2025-12-10"},
            {"order_id": "O101", "customer_id": "C002", "order_date": "2025-12-11"},
            {"order_id": "O102", "customer_id": "C001", "order_date": "2025-12-20"},
        ]
    )
    order_items = pd.DataFrame(
        [
            {"order_id": "O100", "product_id": "P10", "qty": 1},
            {"order_id": "O100", "product_id": "P20", "qty": 2},
            {"order_id": "O101", "product_id": "P30", "qty": 1},
            {"order_id": "O102", "product_id": "P10", "qty": 1},
        ]
    )
    products = pd.DataFrame(
        [
            {"product_id": "P10", "product": "Keyboard", "category_id": "CAT1"},
            {"product_id": "P20", "product": "Mouse", "category_id": "CAT1"},
            {"product_id": "P30", "product": "Monitor", "category_id": "CAT2"},
        ]
    )
    categories = pd.DataFrame(
        [
            {"category_id": "CAT1", "category": "Accessories"},
            {"category_id": "CAT2", "category": "Displays"},
        ]
    )
    shipments = pd.DataFrame(
        [
            {"order_id": "O100", "shipment_id": "S900", "carrier": "DHL"},
            {"order_id": "O101", "shipment_id": "S901", "carrier": "UPS"},
            {"order_id": "O102", "shipment_id": "S902", "carrier": "FedEx"},
        ]
    )
    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "categories": categories,
        "shipments": shipments,
    }


@st.cache_data(show_spinner=False)
def bundled_query() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"order_id": "O100", "customer_id": "C001", "country": "US"},
            {"order_id": "O101", "customer_id": "C002", "country": "SG"},
            {"order_id": "O102", "customer_id": "C001", "country": "US"},
        ]
    )


def read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


@st.cache_data(show_spinner=False)
def load_datalake_from_zip(zip_bytes: bytes) -> Dict[str, pd.DataFrame]:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    tables: Dict[str, pd.DataFrame] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        if not info.filename.lower().endswith(".csv"):
            continue
        name = info.filename.split("/")[-1].rsplit(".", 1)[0].strip()
        if not name:
            continue
        df = read_csv_bytes(zf.read(info))
        if df.empty or df.shape[1] == 0:
            continue
        tables[name] = df
    return tables


def token_for_uploaded_files(files: Sequence) -> Optional[str]:
    if not files:
        return None
    h = hashlib.md5()
    for f in files:
        name = str(getattr(f, "name", "")).encode("utf-8", errors="ignore")
        h.update(name)
        h.update(b"\0")
    return h.hexdigest()


def ensure_nonempty_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cleaned = {}
    for k, df in tables.items():
        if df is None or df.empty or df.shape[1] == 0:
            continue
        df2 = df.copy()
        # Drop fully blank rows (common in CSV exports) so they don't render or affect joins.
        obj_cols = df2.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            df2[obj_cols] = df2[obj_cols].astype("string").apply(lambda s: s.str.strip())
            df2[obj_cols] = df2[obj_cols].replace(r"^\s*$", pd.NA, regex=True)
        df2 = df2.dropna(how="all")
        if df2.empty or df2.shape[1] == 0:
            continue
        cleaned[k] = df2
    return cleaned


def drop_fully_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or df.shape[1] == 0:
        return df
    df2 = df.copy()
    # Normalize whitespace-only strings to NA for object/string columns, then drop rows that
    # are blank across all columns.
    obj_cols = df2.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) > 0:
        df2[obj_cols] = df2[obj_cols].astype("string").apply(lambda s: s.str.strip())
        df2[obj_cols] = df2[obj_cols].replace(r"^\s*$", pd.NA, regex=True)
    df2 = df2.dropna(how="all")
    return df2


def format_path(path: Sequence[str], edges: Sequence[EdgeInfo]) -> str:
    parts = [path[0]]
    for e, nxt in zip(edges, path[1:]):
        parts.append(f"--[{e.key}, {e.score:.2f}]--> {nxt}")
    return " ".join(parts)


def path_to_dot(nodes: Sequence[str], edges: Sequence[EdgeInfo]) -> str:
    lines = [
        "digraph P {",
        '  graph [rankdir="LR", bgcolor="white"];',
        '  node [shape="box", style="rounded,filled", fillcolor="#f7f7f7", color="#444444", fontname="Helvetica"];',
        '  edge [color="#d62728", fontname="Helvetica", penwidth="2.6"];',
    ]
    for n in nodes:
        lines.append(f'  "{n}";')
    for e, nxt in zip(edges, nodes[1:]):
        label = f"{e.key}"
        lines.append(f'  "{e.left}" -> "{nxt}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def token_for_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def simulate_preprocessing(
    steps: Sequence[str],
    seconds_per_step: float = 0.25,
    *,
    container=None,
    log_key: str = "preprocess_log",
) -> None:
    st.session_state[log_key] = []

    scope = container.container() if container is not None else nullcontext()
    with scope:
        progress = st.progress(0, text="Preprocessing...")
        with st.status("Preprocessing", expanded=True) as status:
            for i, step in enumerate(steps, start=1):
                with st.spinner(step):
                    time.sleep(seconds_per_step)
                line = f"✓ {step}"
                st.session_state[log_key].append(line)
                status.write(line)
                progress.progress(int(100 * i / max(len(steps), 1)), text="Preprocessing...")
            status.update(label="Preprocessing complete", state="complete", expanded=False)
        progress.empty()


def render_brand_header() -> None:
    # Simple inline SVG mark for SemDisc (no external assets required).
    svg = """
<svg width="44" height="44" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SemDisc logo">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#4F46E5"/>
      <stop offset="1" stop-color="#06B6D4"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="40" height="40" rx="10" fill="url(#g)"/>
  <circle cx="20" cy="20" r="9" fill="none" stroke="white" stroke-width="3.2"/>
  <line x1="26.5" y1="26.5" x2="33" y2="33" stroke="white" stroke-width="3.2" stroke-linecap="round"/>
  <path d="M16 19c2-3 8-3 10 0" fill="none" stroke="white" stroke-width="2.6" stroke-linecap="round"/>
</svg>
""".strip()

    c1, c2 = st.columns([0.12, 0.88], vertical_alignment="center")
    with c1:
        st.markdown(svg, unsafe_allow_html=True)
    with c2:
        st.markdown(
            "<div style='font-size:34px; font-weight:750; line-height:1.1;'>SemDisc</div>",
            unsafe_allow_html=True,
        )
        st.caption("Semantic discovery of joinable tables and join paths.")


def render_stage_stepper(current_idx: int, stage_labels: Sequence[str]) -> None:
    css = """
<style>
  .sd-stepper { margin-top: 0.75rem; margin-bottom: 0.75rem; }
  .sd-step { display: flex; align-items: center; gap: 0.65rem; }
  .sd-col { display: flex; flex-direction: column; align-items: center; width: 24px; }
  .sd-circle {
    width: 18px; height: 18px; border-radius: 999px;
    border: 2px solid #9CA3AF; background: #ffffff;
    box-sizing: border-box;
  }
  .sd-circle.done { background: #22C55E; border-color: #16A34A; }
  .sd-circle.current {
    background: #22C55E; border-color: #16A34A;
    box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.28), 0 0 12px rgba(34, 197, 94, 0.55);
  }
  .sd-circle.todo { background: #ffffff; border-color: #9CA3AF; }
  .sd-line {
    width: 3px; flex: 1; min-height: 22px;
    background: #D1D5DB; border-radius: 999px;
    margin: 4px 0;
  }
  .sd-line.done { background: #22C55E; }
  .sd-label { font-size: 0.90rem; color: #111827; line-height: 1.1; }
  .sd-label.current { font-weight: 750; }
  .sd-label.todo { color: #4B5563; }
</style>
""".strip()

    lines = ['<div class="sd-stepper">']
    for i, label in enumerate(stage_labels):
        circle_cls = "todo"
        label_cls = "todo"
        if i < current_idx:
            circle_cls = "done"
            label_cls = ""
        elif i == current_idx:
            circle_cls = "current"
            label_cls = "current"

        lines.append('<div class="sd-step">')
        lines.append('<div class="sd-col">')
        lines.append(f'<div class="sd-circle {circle_cls}"></div>')
        if i < len(stage_labels) - 1:
            seg_cls = "done" if i < current_idx else ""
            lines.append(f'<div class="sd-line {seg_cls}"></div>')
        lines.append("</div>")
        lines.append(f'<div class="sd-label {label_cls}">{label}</div>')
        lines.append("</div>")
    lines.append("</div>")

    st.sidebar.markdown(css + "\n" + "\n".join(lines), unsafe_allow_html=True)


def goto_stage(idx: int) -> None:
    st.session_state.page_idx = int(max(0, min(3, idx)))
    st.rerun()

