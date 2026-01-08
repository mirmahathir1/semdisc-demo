from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

import streamlit as st
import streamlit.components.v1 as components


@dataclass(frozen=True)
class HeatmapCell:
    """A single heatmap cell representing one selected join path."""

    path_key: str
    node_count: int
    tuple_count: int


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _red_gradient(value: float) -> str:
    """White -> dark red (single-hue gradient)."""

    t = _clamp(value, 0.0, 1.0)
    # From white to a deep-ish red.
    r = int(round(_lerp(255, 155, t)))
    g = int(round(_lerp(255, 20, t)))
    b = int(round(_lerp(255, 20, t)))
    return _rgb_to_hex(r, g, b)


def compute_bins(values: Sequence[int], num_bins: int = 10) -> List[Tuple[int, int]]:
    """Compute `num_bins` inclusive integer bins between min/max values."""

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if not values:
        return [(0, 0) for _ in range(num_bins)]

    vmin = int(min(values))
    vmax = int(max(values))
    if vmin == vmax:
        return [(vmin, vmax) for _ in range(num_bins)]

    # Create edges as floats, then convert to inclusive integer bins.
    width = (vmax - vmin) / float(num_bins)
    edges = [vmin + i * width for i in range(num_bins + 1)]

    bins: List[Tuple[int, int]] = []
    for i in range(num_bins):
        lo = int(round(edges[i]))
        hi = int(round(edges[i + 1]))
        if i == 0:
            lo = vmin
        if i == num_bins - 1:
            hi = vmax
        if hi < lo:
            hi = lo
        bins.append((lo, hi))

    # Ensure monotonic non-decreasing bins (avoid rounding artifacts).
    fixed: List[Tuple[int, int]] = []
    cur_lo = vmin
    for i, (_lo, hi) in enumerate(bins):
        if i == num_bins - 1:
            fixed.append((cur_lo, vmax))
        else:
            hi2 = max(cur_lo, hi)
            fixed.append((cur_lo, hi2))
            cur_lo = hi2 + 1
    return fixed


def bin_index(value: int, bins: Sequence[Tuple[int, int]]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= value <= hi:
            return i
    # Clamp out-of-range into nearest bin.
    if value < bins[0][0]:
        return 0
    return max(0, len(bins) - 1)


def _get_query_params() -> Dict[str, List[str]]:
    # Streamlit supports both legacy experimental APIs and `st.query_params`.
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        # Convert to "experimental" style dict[str, list[str]]
        out: Dict[str, List[str]] = {}
        for k in qp.keys():
            v = qp.get_all(k) if hasattr(qp, "get_all") else qp.get(k)  # type: ignore[misc]
            if v is None:
                out[k] = []
            elif isinstance(v, list):
                out[k] = [str(x) for x in v]
            else:
                out[k] = [str(v)]
        return out
    except Exception:
        return {k: list(v) for k, v in st.experimental_get_query_params().items()}


def _set_query_params(params: Dict[str, List[str]]) -> None:
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        qp.clear()
        for k, vs in params.items():
            if not vs:
                continue
            qp[k] = vs if len(vs) > 1 else vs[0]
    except Exception:
        st.experimental_set_query_params(**params)


def _with_updated_param(key: str, value: str) -> str:
    params = _get_query_params()
    params[key] = [value]
    flat: List[Tuple[str, str]] = []
    for k, vs in params.items():
        for v in vs:
            flat.append((k, v))
    qs = urlencode(flat)
    return f"?{qs}" if qs else "?"


def consume_query_param(key: str) -> Optional[str]:
    """Read `key` from URL query params, then remove it (best-effort)."""

    params = _get_query_params()
    vals = params.get(key, [])
    if not vals:
        return None
    picked = str(vals[0])
    # Remove it so it doesn't keep re-triggering the selection.
    params.pop(key, None)
    _set_query_params(params)
    return picked


def render_joinpath_heatmap(
    *,
    cells: Sequence[HeatmapCell],
    query_param_key: str,
    title: str = "Join paths heatmap",
    height_px: int = 520,
    selected_best_path_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Render a clickable heatmap as a small Streamlit component.

    - X axis: node_count
    - Y axis: 10 tuple_count bins (computed from `cells`)
    - Color: red gradient by tuple_count (global min/max across cells)
    - Hover: tooltip includes tuple_count and path count
    - Click: returns a dict containing:
        - `best_path_key`: representative path (max tuple_count within cell)
        - `path_keys`: all join paths mapped to the clicked cell
        - `node_count`, `tuple_bin_lo`, `tuple_bin_hi`
    """

    if not cells:
        st.info("No join paths available for heatmap.")
        return None

    # Render via a local static component (avoids full page reloads / new sessions).
    component_dir = os.path.join(os.path.dirname(__file__), "heatmap_component")
    heatmap = components.declare_component("joinpath_heatmap", path=component_dir)

    payload = [
        {
            "path_key": c.path_key,
            "node_count": int(c.node_count),
            "tuple_count": int(c.tuple_count),
        }
        for c in cells
    ]

    # Note: we keep `query_param_key` in the function signature for backwards
    # compatibility, but selection is handled via component value (no query params).
    picked: Any = heatmap(
        cells=payload,
        title=title,
        height_px=int(height_px),
        selected_best_path_key=selected_best_path_key,
        key=f"heatmap_{query_param_key}",
        default=None,
    )

    if isinstance(picked, dict):
        # New payload includes `path_keys` + `best_path_key`. Older payload may include `path_key`.
        if "path_key" in picked and "best_path_key" not in picked:
            picked = {**picked, "best_path_key": picked.get("path_key")}
        return picked
    if isinstance(picked, str) and picked:
        # Backwards compatibility if component returns a raw string.
        return {"best_path_key": picked, "path_keys": [picked]}
    return None

