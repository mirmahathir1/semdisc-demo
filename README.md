## Streamlit demo

### Run

```bash
cd semdisc-demo
python -m pip install -r requirements.txt
streamlit run app.py
```

### What you can do

- **Upload a data lake folder** as a single ZIP containing CSV files (each CSV becomes a table)
- **Input a query table** by uploading a CSV
- **Get top‑k join paths** (ranked), visualize the join graph, and **materialize** the selected path

### `app.py` (Streamlit + Graphviz)

```python
from __future__ import annotations

import io
import itertools
import hashlib
import time
import zipfile
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


def _ensure_nonempty_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cleaned = {}
    for k, df in tables.items():
        if df is None or df.empty or df.shape[1] == 0:
            continue
        df2 = df.copy()
        obj_cols = df2.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            df2[obj_cols] = df2[obj_cols].astype("string").apply(lambda s: s.str.strip())
            df2[obj_cols] = df2[obj_cols].replace(r"^\s*$", pd.NA, regex=True)
        df2 = df2.dropna(how="all")
        if df2.empty or df2.shape[1] == 0:
            continue
        cleaned[k] = df2
    return cleaned


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


def _simulate_preprocessing(steps: Sequence[str], seconds_per_step: float = 0.25) -> None:
    progress = st.progress(0, text="Preprocessing...")
    with st.status("Preprocessing", expanded=True) as status:
        for i, step in enumerate(steps, start=1):
            with st.spinner(step):
                time.sleep(seconds_per_step)
            status.write(f"✓ {step}")
            progress.progress(int(100 * i / max(len(steps), 1)), text="Preprocessing...")
        status.update(label="Preprocessing complete", state="complete", expanded=False)
    progress.empty()


def _render_brand_header() -> None:
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

    # Always keep a prepopulated query table available by default.
    if st.session_state.query_df is None:
        st.session_state.query_df = _bundled_query()
        st.session_state.query_ready = False
        st.session_state.query_submitted = False

    # Left bar: navigation only (radio buttons)
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "Upload and Preprocess",
            "Navigate Processed Data",
            "Input Query Table",
            "Retrieved Join Paths",
        ],
        label_visibility="collapsed",
    )

    if page == "Upload and Preprocess":
        st.subheader("Upload and Preprocess")
        zip_file = st.file_uploader("Data lake (ZIP of CSV files)", type=["zip"])

        st.subheader("Hyperparameters")
        c3, c4 = st.columns(2)
        with c3:
            st.session_state.hp_simhash_size = st.slider(
                "SimHash size (bits)",
                min_value=32,
                max_value=512,
                value=int(st.session_state.hp_simhash_size),
                step=32,
            )
            st.session_state.hp_num_hashes_minhash = st.slider(
                "Number of hashes for MinHash",
                min_value=16,
                max_value=512,
                value=int(st.session_state.hp_num_hashes_minhash),
                step=16,
            )
            st.session_state.hp_simple_path_max_length = st.slider(
                "Simple path max length",
                min_value=1,
                max_value=5,
                value=int(st.session_state.hp_simple_path_max_length),
                step=1,
            )
        with c4:
            st.session_state.hp_join_edge_threshold = st.slider(
                "Join edge threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.hp_join_edge_threshold),
                step=0.01,
            )
            st.session_state.hp_semantic_type_similarity_threshold = st.slider(
                "Semantic type similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.hp_semantic_type_similarity_threshold),
                step=0.01,
            )

        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            preprocess_clicked = st.button("Preprocess", type="primary", use_container_width=True)
        with col2:
            reset_clicked = st.button("Reset processed data", use_container_width=True)

        if reset_clicked:
            st.session_state.processed_ready = False
            st.session_state.tables = {}
            st.session_state.datalake_token = None

        uploaded_zip_bytes: Optional[bytes] = zip_file.getvalue() if zip_file is not None else None
        uploaded_token: Optional[str] = (
            _token_for_bytes(uploaded_zip_bytes) if uploaded_zip_bytes is not None else None
        )

        if uploaded_token is not None and st.session_state.datalake_token != uploaded_token:
            # New upload selected; invalidate prior processed results.
            st.session_state.processed_ready = False
            st.session_state.tables = {}
            st.session_state.datalake_token = uploaded_token

        if preprocess_clicked:
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
            _simulate_preprocessing(steps=steps)

            if uploaded_zip_bytes is not None:
                tables = _load_datalake_from_zip(uploaded_zip_bytes)
            else:
                tables = _bundled_datalake()
            tables = _ensure_nonempty_tables(tables)
            if not tables:
                st.error("No tables found in the data lake.")
                return
            st.session_state.tables = tables
            st.session_state.processed_ready = True
            st.session_state.nav_min_edge_score = float(st.session_state.hp_join_edge_threshold)
            st.session_state.nav_max_hops = int(st.session_state.hp_simple_path_max_length)
            st.session_state.ret_min_edge_score = float(st.session_state.hp_join_edge_threshold)
            st.session_state.ret_max_hops = int(st.session_state.hp_simple_path_max_length)

        if st.session_state.processed_ready:
            st.success(f"Processed data ready: {len(st.session_state.tables)} tables.")
        else:
            st.info("Click **Preprocess** to load and prepare the data lake.")

        return

    if page == "Navigate Processed Data":
        st.subheader("Navigate Processed Data")
        if not st.session_state.processed_ready:
            st.info("Go to **Upload and Preprocess** first.")
            return

        tables: Dict[str, pd.DataFrame] = st.session_state.tables
        st.write(f"{len(tables)} tables available.")
        nav_min_edge_score = float(st.session_state.nav_min_edge_score)
        nav_max_hops = int(st.session_state.nav_max_hops)
        nav_top_k_per_table = 10

        graph = _build_join_graph(tables, min_edge_score=nav_min_edge_score)

        top_left, top_right = st.columns([0.52, 0.48], gap="large")

        with top_left:
            st.subheader("Table preview")
            table_name = st.selectbox("Table", options=sorted(tables.keys()))
            df = tables[table_name]
            st.write(f"Rows: {df.shape[0]} • Cols: {df.shape[1]}")
            st.dataframe(df, use_container_width=True, height=420)

        with top_right:
            st.subheader("Join graph")
            st.graphviz_chart(_to_dot(graph), use_container_width=True)
            st.caption("Edge labels show the join key and a join-strength score.")

        st.divider()
        st.subheader("Join path index")
        if not graph:
            st.warning("No join edges found with the current settings.")
            return

        start_table = st.selectbox(
            "Start table",
            options=sorted(tables.keys()),
            help="Paths are enumerated from this start table.",
            key="nav_start_table",
        )

        paths = _simple_paths_topk(
            graph=graph,
            start=start_table,
            max_hops=nav_max_hops,
            k=nav_top_k_per_table,
        )

        if not paths:
            st.warning("No join paths found from the selected start table.")
            return

        rows = []
        for rank, (nodes, score, edges) in enumerate(paths, start=1):
            if len(nodes) <= 1:
                continue
            rows.append(
                {
                    "rank": rank,
                    "start": nodes[0],
                    "end": nodes[-1],
                    "hops": len(nodes) - 1,
                    "path": " -> ".join(nodes),
                    "join_keys": ", ".join([e.key for e in edges]) if edges else "",
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)

        st.session_state.setdefault("nav_path_graph_rank", None)
        st.markdown("**Join Paths**")
        for rank, (nodes, _score, edges) in enumerate(paths, start=1):
            if len(nodes) <= 1:
                continue
            label = f"#{rank}: {nodes[0]} → {nodes[-1]}  ({len(nodes)-1} hops)"
            c_btn, c_txt = st.columns([0.22, 0.78], vertical_alignment="center")
            with c_btn:
                if st.button("Show join path", key=f"nav_show_path_{rank}"):
                    st.session_state.nav_path_graph_rank = rank
            with c_txt:
                st.code(" -> ".join(nodes), language="text")

        if st.session_state.nav_path_graph_rank is not None:
            r = int(st.session_state.nav_path_graph_rank)
            if 1 <= r <= len(paths):
                nodes, _score, edges = paths[r - 1]
                st.subheader(f"Join Path (#{r})")
                st.graphviz_chart(_path_to_dot(nodes, edges), use_container_width=True)
        return

    if page == "Input Query Table":
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
        return

    # Retrieved Join Paths
    st.subheader("Retrieved Join Paths")
    if not st.session_state.processed_ready:
        st.info("Go to **Upload and Preprocess** first.")
        return
    if not st.session_state.query_submitted or st.session_state.query_df is None:
        st.info("Go to **Input Query Table** first.")
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


if __name__ == "__main__":
    main()
```

