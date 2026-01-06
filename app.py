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


def _normalize_cell(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = {x for x in a if x}
    sb = {x for x in b if x}
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _best_join_key_and_score(left: pd.DataFrame, right: pd.DataFrame) -> Optional[Tuple[str, float]]:
    common_cols = [c for c in left.columns if c in right.columns]
    if not common_cols:
        return None

    best = None
    for col in common_cols:
        lvals = (_normalize_cell(v) for v in left[col].head(5000))
        rvals = (_normalize_cell(v) for v in right[col].head(5000))
        score = _jaccard(lvals, rvals)
        if best is None or score > best[1]:
            best = (col, score)
    return best


def _build_join_graph(tables: Dict[str, pd.DataFrame], min_edge_score: float) -> Dict[str, List[EdgeInfo]]:
    graph: Dict[str, List[EdgeInfo]] = defaultdict(list)
    names = sorted(tables.keys())
    for a, b in itertools.combinations(names, 2):
        best = _best_join_key_and_score(tables[a], tables[b])
        if best is None:
            continue
        key, score = best
        if score < min_edge_score:
            continue
        graph[a].append(EdgeInfo(left=a, right=b, key=key, score=score))
        graph[b].append(EdgeInfo(left=b, right=a, key=key, score=score))
    return graph


def _attach_query_node(
    graph: Dict[str, List[EdgeInfo]],
    tables: Dict[str, pd.DataFrame],
    query: pd.DataFrame,
    min_edge_score: float,
) -> Tuple[Dict[str, List[EdgeInfo]], Dict[str, pd.DataFrame]]:
    tables2 = dict(tables)
    tables2["query"] = query
    g2: Dict[str, List[EdgeInfo]] = defaultdict(list, {k: list(v) for k, v in graph.items()})

    for name, df in tables.items():
        best = _best_join_key_and_score(query, df)
        if best is None:
            continue
        key, score = best
        if score < min_edge_score:
            continue
        g2["query"].append(EdgeInfo(left="query", right=name, key=key, score=score))
        g2[name].append(EdgeInfo(left=name, right="query", key=key, score=score))

    return g2, tables2


def _simple_paths_topk(
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


def _materialize_path(
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


def _to_dot(
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
def _bundled_datalake() -> Dict[str, pd.DataFrame]:
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
def _bundled_query() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"order_id": "O100", "customer_id": "C001", "country": "US"},
            {"order_id": "O101", "customer_id": "C002", "country": "SG"},
            {"order_id": "O102", "customer_id": "C001", "country": "US"},
        ]
    )


def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


@st.cache_data(show_spinner=False)
def _load_datalake_from_zip(zip_bytes: bytes) -> Dict[str, pd.DataFrame]:
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
        df = _read_csv_bytes(zf.read(info))
        if df.empty or df.shape[1] == 0:
            continue
        tables[name] = df
    return tables


def _token_for_uploaded_files(files: Sequence) -> Optional[str]:
    if not files:
        return None
    h = hashlib.md5()
    for f in files:
        name = str(getattr(f, "name", "")).encode("utf-8", errors="ignore")
        h.update(name)
        h.update(b"\0")
    return h.hexdigest()


def _ensure_nonempty_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
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


def _drop_fully_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
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


def _format_path(path: Sequence[str], edges: Sequence[EdgeInfo]) -> str:
    parts = [path[0]]
    for e, nxt in zip(edges, path[1:]):
        parts.append(f"--[{e.key}, {e.score:.2f}]--> {nxt}")
    return " ".join(parts)


def _path_to_dot(nodes: Sequence[str], edges: Sequence[EdgeInfo]) -> str:
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


def _token_for_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _simulate_preprocessing(
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


def _render_brand_header() -> None:
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


def _render_stage_stepper(current_idx: int, stage_labels: Sequence[str]) -> None:
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


def _goto_stage(idx: int) -> None:
    st.session_state.page_idx = int(max(0, min(3, idx)))
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="SemDisc", layout="wide")
    _render_brand_header()

    # Session state
    st.session_state.setdefault("processed_ready", False)
    st.session_state.setdefault("datalake_token", None)
    st.session_state.setdefault("tables", {})
    st.session_state.setdefault("query_ready", False)
    st.session_state.setdefault("query_df", None)
    st.session_state.setdefault("query_submitted", False)
    # Hyperparameters (mirrors `dice/stages/offline_pipeline_cli.py` CLI flags)
    st.session_state.setdefault("hp_simhash_size", 128)
    st.session_state.setdefault("hp_num_hashes_minhash", 128)
    st.session_state.setdefault("hp_join_edge_threshold", 0.10)
    st.session_state.setdefault("hp_semantic_type_similarity_threshold", 0.70)
    st.session_state.setdefault("hp_simple_path_max_length", 3)
    # Defaults for downstream screens (seeded from preprocessing hyperparameters)
    st.session_state.setdefault("nav_min_edge_score", float(st.session_state.hp_join_edge_threshold))
    st.session_state.setdefault("nav_max_hops", int(st.session_state.hp_simple_path_max_length))
    st.session_state.setdefault("ret_min_edge_score", float(st.session_state.hp_join_edge_threshold))
    st.session_state.setdefault("ret_max_hops", int(st.session_state.hp_simple_path_max_length))
    st.session_state.setdefault("ret_top_k", 10)
    st.session_state.setdefault("page_idx", 0)
    st.session_state.setdefault("preprocess_log", [])

    # Always keep a prepopulated query table available by default.
    if st.session_state.query_df is None:
        st.session_state.query_df = _bundled_query()
        st.session_state.query_ready = False
        st.session_state.query_submitted = False

    stage_labels = [
        "Upload & Preprocess",
        "Navigate Processed Data",
        "Input Query Table",
        "Retrieved Join Paths",
    ]
    page_idx = int(st.session_state.page_idx)
    page_idx = max(0, min(3, page_idx))
    st.session_state.page_idx = page_idx
    _render_stage_stepper(current_idx=page_idx, stage_labels=stage_labels)

    if page_idx == 0:
        left_col, right_col = st.columns([0.62, 0.38], gap="large")

        with right_col:
            st.subheader("Preprocessing updates")
            updates_placeholder = st.empty()

            def _render_preprocess_updates() -> None:
                log = list(st.session_state.get("preprocess_log", []) or [])
                with updates_placeholder.container():
                    if log:
                        st.code("\n".join(log[-200:]), language="text")
                    else:
                        st.caption("No preprocessing run yet.")

            _render_preprocess_updates()

        with left_col:
            st.subheader("Upload and Preprocess")
            uploaded_files = st.file_uploader(
                "Data lake folder (upload CSV files)",
                type=["csv"],
                accept_multiple_files=True,
                help="Streamlit can't directly upload a folder. Upload any CSV file(s) from the folder to unlock preprocessing (the demo run uses the bundled sample data).",
            )

            st.subheader("Hyperparameters")
            c3, c4, c5 = st.columns(3)
            with c3:
                st.session_state.hp_simhash_size = st.slider(
                    "SimHash size (bits)",
                    min_value=32,
                    max_value=512,
                    value=int(st.session_state.hp_simhash_size),
                    step=32,
                )
                st.session_state.hp_join_edge_threshold = st.slider(
                    "Join edge threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.hp_join_edge_threshold),
                    step=0.01,
                )
            with c4:
                st.session_state.hp_num_hashes_minhash = st.slider(
                    "Number of hashes for MinHash",
                    min_value=16,
                    max_value=512,
                    value=int(st.session_state.hp_num_hashes_minhash),
                    step=16,
                )
                st.session_state.hp_semantic_type_similarity_threshold = st.slider(
                    "Semantic type similarity threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.hp_semantic_type_similarity_threshold),
                    step=0.01,
                )
            with c5:
                st.session_state.hp_simple_path_max_length = st.slider(
                    "Simple path max length",
                    min_value=1,
                    max_value=5,
                    value=int(st.session_state.hp_simple_path_max_length),
                    step=1,
                )

            preprocess_disabled = not uploaded_files
            if preprocess_disabled:
                st.info("Upload your data lake CSV files to enable preprocessing.")

            preprocess_clicked = st.button(
                "Preprocess",
                type="primary",
                use_container_width=True,
                disabled=preprocess_disabled,
            )
            reset_clicked = st.button("Reset processed data", use_container_width=True)

        if reset_clicked:
            st.session_state.processed_ready = False
            st.session_state.tables = {}
            st.session_state.datalake_token = None

        uploaded_token: Optional[str] = _token_for_uploaded_files(uploaded_files)

        if uploaded_token is not None and st.session_state.datalake_token != uploaded_token:
            # New upload selected; invalidate prior processed results.
            st.session_state.processed_ready = False
            st.session_state.tables = {}
            st.session_state.datalake_token = uploaded_token

        if preprocess_clicked:
            # Keep ordering aligned to `offline_pipeline_cli.py` routines (0..12),
            # but only simulate the main milestones.
            routine_steps = {
                0: "Deleting processed data",
                1: "Copying data lake",
                2: "Sanitizing tables",
                3: "Extracting semantic types",
                4: "Computing table embeddings",
                6: "Computing MinHash similarities",
                7: "Computing semantic hashes",
                8: "Computing diversity index",
                9: "Computing semantic type similarities",
                10: "Detecting numeric columns",
                11: "Building join graph",
                12: "Building join path index",
            }
            steps = [routine_steps[i] for i in range(0, 13) if i in routine_steps]
            _simulate_preprocessing(steps=steps, container=updates_placeholder)
            _render_preprocess_updates()

            if not uploaded_files:
                st.warning("Please upload your data lake CSV files before preprocessing.")
                return
            tables = _bundled_datalake()
            tables = _ensure_nonempty_tables(tables)
            if not tables:
                st.error("No tables found in the data lake.")
                return
            st.session_state.tables = tables
            st.session_state.processed_ready = True
            # Seed downstream defaults from preprocessing hyperparameters.
            st.session_state.nav_min_edge_score = float(st.session_state.hp_join_edge_threshold)
            st.session_state.nav_max_hops = int(st.session_state.hp_simple_path_max_length)
            st.session_state.ret_min_edge_score = float(st.session_state.hp_join_edge_threshold)
            st.session_state.ret_max_hops = int(st.session_state.hp_simple_path_max_length)

        if st.session_state.processed_ready:
            st.success(f"Processed data ready: {len(st.session_state.tables)} tables.")
        else:
            st.info("Click **Preprocess** to load and prepare the data lake.")

        if st.session_state.processed_ready:
            if st.button("Proceed to next stage", type="primary", use_container_width=True):
                _goto_stage(1)

        return

    if page_idx == 1:
        st.subheader("Navigate Processed Data")
        if not st.session_state.processed_ready:
            st.info("Go to **Upload and Preprocess** first.")
            c_nav1, c_nav2 = st.columns([0.45, 0.55])
            with c_nav1:
                if st.button("Back to previous stage", use_container_width=True):
                    _goto_stage(0)
            with c_nav2:
                if st.button("Proceed to next stage", type="primary", use_container_width=True):
                    st.warning("Please run **Preprocess** before proceeding.")
            return

        tables: Dict[str, pd.DataFrame] = st.session_state.tables
        st.write(f"{len(tables)} tables available.")
        nav_min_edge_score = float(st.session_state.nav_min_edge_score)
        nav_max_hops = int(st.session_state.nav_max_hops)
        nav_top_k_per_table = 10

        graph = _build_join_graph(tables, min_edge_score=nav_min_edge_score)

        # Compute "all" join paths by enumerating top-k paths from every start table,
        # then deduplicate and globally sort by score.
        paths: List[Tuple[List[str], float, List[EdgeInfo]]] = []
        if graph:
            by_nodes: Dict[Tuple[str, ...], Tuple[List[str], float, List[EdgeInfo]]] = {}
            for start_table in sorted(tables.keys()):
                for nodes, score, edges in _simple_paths_topk(
                    graph=graph,
                    start=start_table,
                    max_hops=nav_max_hops,
                    k=nav_top_k_per_table,
                ):
                    if len(nodes) <= 1:
                        continue
                    key = tuple(nodes)
                    prev = by_nodes.get(key)
                    if prev is None or score > prev[1]:
                        by_nodes[key] = (nodes, score, edges)
            paths = sorted(by_nodes.values(), key=lambda x: x[1], reverse=True)

        top_left, top_right = st.columns([0.52, 0.48], gap="large")

        with top_left:
            st.subheader("Table preview")
            table_name = st.selectbox("Table", options=sorted(tables.keys()))
            df = _drop_fully_blank_rows(tables[table_name])
            st.write(f"Rows: {df.shape[0]} • Cols: {df.shape[1]}")
            # Show ~5 rows worth of height; remaining rows are scrollable within the table.
            header_px = 36
            row_px = 35
            preview_rows = 5
            table_height = header_px + row_px * max(1, min(preview_rows, int(df.shape[0])))
            st.dataframe(df, use_container_width=True, height=table_height)

        with top_right:
            st.subheader("Join paths")
            if not graph:
                st.warning("No join edges found with the current settings.")
            elif not paths:
                st.warning("No join paths found with the current settings.")
            else:
                st.markdown(
                    """
<style>
  .sd-pathcard { border: 1px solid #E5E7EB; background: #FFFFFF; border-radius: 12px; padding: 10px 12px; margin-bottom: 10px; }
  .sd-pathhead { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
  .sd-rank { font-weight: 750; color: #111827; }
  .sd-score { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; color: #111827; }
  .sd-meta { font-size: 0.86rem; color: #374151; margin-top: 6px; }
  .sd-path { margin-top: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.86rem; color: #111827; background: #F9FAFB; border: 1px solid #F3F4F6; border-radius: 10px; padding: 8px; }
  .sd-pathgraph-container { height: 320px; overflow: auto; margin-top: 16px; }
  .sd-pathgraph-container div[data-testid="stGraphVizChart"] { height: 320px; overflow: auto; }
  .sd-pathgraph-container div[data-testid="stGraphVizChart"] svg { height: 320px !important; width: auto !important; }
</style>
""",
                    unsafe_allow_html=True,
                )

                # Render a per-path button + HTML card in a loop (no dropdown).
                st.session_state.setdefault("nav_selected_path_idx", 0)
                selected_idx = int(st.session_state.nav_selected_path_idx or 0)
                selected_idx = max(0, min(selected_idx, len(paths) - 1))
                st.session_state.nav_selected_path_idx = selected_idx

                # CSS for styling path cards
                st.markdown(
                    """
<style>
  .sd-pathcard { border: 1px solid #E5E7EB; background: #FFFFFF; border-radius: 12px; padding: 10px 12px; margin-bottom: 10px; }
  .sd-pathhead { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
  .sd-rank { font-weight: 750; color: #111827; }
  .sd-score { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; color: #111827; }
  .sd-meta { font-size: 0.86rem; color: #374151; margin-top: 6px; }
  .sd-path { margin-top: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.86rem; color: #111827; background: #F9FAFB; border: 1px solid #F3F4F6; border-radius: 10px; padding: 8px; }
</style>
""",
                    unsafe_allow_html=True,
                )
                
                # Use Streamlit's native scrollable container (height parameter)
                with st.container(height=480, border=True):
                    for idx, (nodes, score, edges) in enumerate(paths):
                        if len(nodes) <= 1:
                            continue
                        rank = idx + 1
                        hops = len(nodes) - 1
                        join_keys = ", ".join([e.key for e in edges]) if edges else ""
                        path_txt = " -> ".join(nodes)
                        
                        bcol, ccol = st.columns([0.22, 0.78], vertical_alignment="top")
                        with bcol:
                            label = "Showing" if idx == selected_idx else "Show join path"
                            if st.button(label, key=f"nav_show_join_path_{rank}", use_container_width=True):
                                st.session_state.nav_selected_path_idx = idx
                                st.rerun()
                        with ccol:
                            card_bg = "#EFF6FF" if idx == selected_idx else "#FFFFFF"
                            card_html = "\n".join(
                                [
                                    f'<div class="sd-pathcard" style="background: {card_bg};">',
                                    '<div class="sd-pathhead">',
                                    f'<div class="sd-rank">#{rank} • {nodes[0]} → {nodes[-1]} • {hops} hop{"s" if hops != 1 else ""}</div>',
                                    f'<div class="sd-score">{float(score):.4f}</div>',
                                    "</div>",
                                    f'<div class="sd-meta"><b>Join keys:</b> {join_keys if join_keys else "—"}</div>',
                                    f'<div class="sd-path">{path_txt}</div>',
                                    "</div>",
                                ]
                            )
                            st.markdown(card_html, unsafe_allow_html=True)

                # Render selected path graph
                nodes, _score, edges = paths[selected_idx]
                st.subheader(f"Path graph (#{selected_idx + 1})")
                st.markdown('<div class="sd-pathgraph-container">', unsafe_allow_html=True)
                st.graphviz_chart(_path_to_dot(nodes, edges), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        if st.button("Proceed to next stage", type="primary", use_container_width=True):
            _goto_stage(2)
        return

    if page_idx == 2:
        st.subheader("Input Query Table")
        c1, c2 = st.columns([0.5, 0.5])
        with c1:
            submit_query = st.button("Submit query table", type="primary", use_container_width=True)
        with c2:
            reset_query = st.button("Reset query table", use_container_width=True)

        if reset_query:
            st.session_state.query_df = _bundled_query()
            st.session_state.query_ready = False
            st.session_state.query_submitted = False

        if submit_query:
            st.session_state.query_ready = True
            st.session_state.query_submitted = True
            st.success("Query table submitted. Proceed to **Retrieved Join Paths**.")

        qdf: pd.DataFrame = st.session_state.query_df
        st.success(f"Query table ready: Rows {qdf.shape[0]} • Cols {qdf.shape[1]}")
        st.dataframe(qdf.head(200), use_container_width=True, height=420)
        if not st.session_state.query_submitted:
            st.info("Click **Submit query table**, then proceed to **Retrieved Join Paths**.")
        c_nav1, c_nav2 = st.columns([0.45, 0.55])
        with c_nav1:
            if st.button("Back to previous stage", use_container_width=True):
                _goto_stage(1)
        with c_nav2:
            if st.button("Proceed to next stage", type="primary", use_container_width=True):
                if not st.session_state.query_submitted:
                    st.warning("Please **Submit query table** before proceeding.")
                else:
                    _goto_stage(3)
        return

    # Retrieved Join Paths
    st.subheader("Retrieved Join Paths")
    if not st.session_state.processed_ready:
        st.info("Go to **Upload and Preprocess** first.")
        if st.button("Back to previous stage", use_container_width=True):
            _goto_stage(0)
        return
    if not st.session_state.query_submitted or st.session_state.query_df is None:
        st.info("Go to **Input Query Table** first.")
        if st.button("Back to previous stage", use_container_width=True):
            _goto_stage(2)
        return

    tables = st.session_state.tables
    query_df: pd.DataFrame = st.session_state.query_df

    st.markdown("**Settings**")
    c1 = st.columns(1)[0]
    with c1:
        top_k = st.slider(
            "Top‑k join paths",
            min_value=1,
            max_value=50,
            value=int(st.session_state.ret_top_k),
            step=1,
            key="ret_top_k",
        )
    max_hops = int(st.session_state.ret_max_hops)
    min_edge_score = float(st.session_state.ret_min_edge_score)

    graph = _build_join_graph(tables, min_edge_score=min_edge_score)
    graph, tables_with_query = _attach_query_node(
        graph=graph, tables=tables, query=query_df, min_edge_score=min_edge_score
    )
    paths = _simple_paths_topk(graph=graph, start="query", max_hops=max_hops, k=top_k)

    left, right = st.columns([0.55, 0.45], gap="large")
    with right:
        st.subheader("Join graph")
        st.graphviz_chart(_to_dot(graph), use_container_width=True)

    with left:
        st.subheader("Top join paths")
        if not paths:
            st.warning("No join paths found. Try adjusting preprocessing hyperparameters and re-running preprocessing.")
            return

        rows = []
        for idx, (nodes, score, edges) in enumerate(paths, start=1):
            rows.append(
                {
                    "rank": idx,
                    "score": round(float(score), 4),
                    "path": " -> ".join(nodes),
                    "joins": ", ".join([e.key for e in edges]) if edges else "",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=240)

        labels = [
            f"#{i+1}  {paths[i][1]:.4f}  |  {paths[i][0][-1]}  |  {paths[i][0]}"
            for i in range(len(paths))
        ]
        pick = st.selectbox(
            "Select a path to materialize",
            options=list(range(len(paths))),
            format_func=lambda i: labels[i],
        )

    selected_nodes, selected_score, selected_edges = paths[pick]

    st.divider()
    st.subheader("Selected path")
    st.write(f"**Score:** {selected_score:.4f}")
    st.code(_format_path(selected_nodes, selected_edges), language="text")

    st.subheader("Highlighted path in join graph")
    highlight = [(e.left, e.right) for e in selected_edges]
    st.graphviz_chart(_to_dot(graph, highlight_edges=highlight), use_container_width=True)

    st.subheader("Materialized table")
    try:
        materialized = _materialize_path(
            tables=tables_with_query, path_nodes=selected_nodes, path_edges=selected_edges
        )
    except Exception as e:
        st.error(f"Failed to materialize the selected path: {e}")
        return

    st.write(f"Rows: {materialized.shape[0]} • Cols: {materialized.shape[1]}")
    st.dataframe(materialized.head(200), use_container_width=True, height=420)

    csv_bytes = materialized.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download materialized CSV",
        data=csv_bytes,
        file_name="materialized.csv",
        mime="text/csv",
    )
    st.divider()
    c_nav1, c_nav2 = st.columns([0.45, 0.55])
    with c_nav1:
        if st.button("Back to previous stage", use_container_width=True):
            _goto_stage(2)
    with c_nav2:
        if st.button("Proceed to next stage (restart)", type="primary", use_container_width=True):
            _goto_stage(0)


if __name__ == "__main__":
    main()


