"""Column join-weight scoring via SBERT -> SimHash -> MinHash -> Jaccard.

This mirrors the dice reference pipeline:

* ``semantic_hash_index.py``: encode column values with SBERT, project them
  through shared random hyperplanes into SimHash bit-strings, then build one
  MinHash per column over those strings (``compute_minhash`` /
  ``extract_simhash_using_hyperplanes`` / ``jaccard_similarity``).
* ``precompute_all_join_paths_minhashes.py``: an edge weight between two
  columns is ``minhash_a.jaccard(minhash_b)``.

The SBERT backend is the default. A ``raw`` backend (MinHash directly over the
cell strings, like ``column_min_hashes``) is used as a fast, deterministic
fallback for tests or when sentence-transformers / the model is unavailable.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from datasketch import MinHash

logger = logging.getLogger(__name__)

# The model dice uses (text_encoder.SBERT_CARD).
SBERT_MODEL = os.getenv("SEMDISC_SBERT_MODEL", "paraphrase-distilroberta-base-v1")
# "sbert" (default) or "raw". Auto-falls back to "raw" if SBERT can't load.
_BACKEND = os.getenv("SEMDISC_SEMANTIC_BACKEND", "sbert").strip().lower()
_HYPERPLANE_SEED = 0
_MINHASH_SEED = 1
# Bump when the cached object schema changes so old caches are ignored.
_CACHE_VERSION = "v2-colsig"

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "semdisc_minhash"
_MEMO: Dict[str, Dict[str, Dict[str, Optional[MinHash]]]] = {}
_model = None
_model_failed = False


@dataclass
class ColumnSignature:
    """One column's MinHash plus how it was built.

    Numeric columns -> MinHash over raw values (equi-join, no embedding).
    Text columns    -> MinHash over SBERT SimHash strings (semantic join).
    """

    minhash: Optional[MinHash]
    is_numeric: bool


def _column_values(df: pd.DataFrame, col: str) -> list[str]:
    """Distinct non-null cell values as strings (MinHash is a set sketch, so
    deduping is equivalent to dice updating once per row)."""
    series = df[col].dropna()
    return [str(v) for v in pd.unique(series)]


def is_numeric_column(df: pd.DataFrame, col: str, values: list[str]) -> bool:
    """A column is numeric if its pandas dtype is numeric, or if more than
    half of its non-null values parse as numbers (dice: numeric_portion > 0.5)."""
    if pd.api.types.is_numeric_dtype(df[col]):
        return True
    if not values:
        return False
    parsed = 0
    for v in values:
        try:
            float(str(v).replace(",", "").strip())
            parsed += 1
        except (TypeError, ValueError):
            pass
    return parsed / len(values) > 0.5


def _get_model():
    global _model, _model_failed
    if _model is not None:
        return _model
    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer(SBERT_MODEL, device="cpu")
    return _model


def _hyperplanes(simhash_size: int, dim: int) -> np.ndarray:
    # Shared across every column and deterministic so cached minhashes stay
    # comparable across runs (dice shares one hyperplane set per build).
    rng = np.random.default_rng(_HYPERPLANE_SEED)
    return rng.standard_normal((simhash_size, dim))


def extract_simhash(embeddings: np.ndarray, hyperplanes: np.ndarray) -> list[str]:
    """Sign of each embedding projected onto the hyperplanes -> bit string
    (dice: ``extract_simhash_using_hyperplanes``)."""
    bits = (np.dot(embeddings, hyperplanes.T) > 0).astype(int)
    return ["".join(map(str, row)) for row in bits]


def compute_minhash(signature: list[str], num_perm: int) -> Optional[MinHash]:
    """MinHash over a list of strings (dice: ``compute_minhash``)."""
    if not signature:
        return None
    m = MinHash(num_perm=num_perm, seed=_MINHASH_SEED)
    for s in signature:
        if s is not None:
            m.update(s.encode("utf8"))
    return m


def jaccard(a: Optional[MinHash], b: Optional[MinHash]) -> float:
    """Edge weight = estimated Jaccard between two column MinHashes."""
    if a is None or b is None:
        return 0.0
    return float(a.jaccard(b))


def _intended_backend() -> str:
    """Backend without touching the model, so cache hits stay fast."""
    if _BACKEND == "raw" or _model_failed:
        return "raw"
    return "sbert"


_TYPE_EMB: Dict[str, np.ndarray] = {}


def _embed_types(types: list[str]) -> Dict[str, np.ndarray]:
    missing = [t for t in types if t not in _TYPE_EMB]
    if missing:
        model = _get_model()
        vecs = model.encode(
            missing, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        for t, v in zip(missing, vecs):
            _TYPE_EMB[t] = v
    return {t: _TYPE_EMB[t] for t in types}


def semantic_type_cosine(type_a: str, type_b: str) -> Optional[float]:
    """Cosine similarity between the SBERT embeddings of two semantic-type
    strings. Returns ``None`` when SBERT is unavailable so the caller can fall
    back (e.g. exact type-string match)."""
    global _model_failed
    a = (type_a or "").strip()
    b = (type_b or "").strip()
    if not a or not b:
        return None
    if _intended_backend() != "sbert":
        return None
    try:
        emb = _embed_types([a, b])
    except Exception as exc:  # pragma: no cover - environment dependent
        _model_failed = True
        logger.warning("SBERT unavailable (%s); semantic-type cosine disabled.", exc)
        return None
    # Embeddings are L2-normalized, so the dot product is the cosine.
    return float(np.dot(emb[a], emb[b]))


def _fingerprint(tables: Dict[str, pd.DataFrame], simhash_size: int, num_perm: int, backend: str) -> str:
    parts = [_CACHE_VERSION, backend, str(simhash_size), str(num_perm)]
    for name in sorted(tables):
        df = tables[name]
        col_hash = int(pd.util.hash_pandas_object(df, index=False).sum() & 0xFFFFFFFFFFFF)
        parts.append(f"{name}:{','.join(map(str, df.columns))}:{df.shape}:{col_hash}")
    import hashlib

    return hashlib.sha1("|".join(parts).encode("utf8")).hexdigest()


def _build(tables: Dict[str, pd.DataFrame], simhash_size: int, num_perm: int, backend: str):
    minhashes: Dict[str, Dict[str, ColumnSignature]] = {}
    model = _get_model() if backend == "sbert" else None
    hyperplanes = None  # built lazily from the first embedding's dimension

    for name, df in tables.items():
        minhashes[name] = {}
        for col in map(str, df.columns):
            values = _column_values(df, col)
            numeric = is_numeric_column(df, col, values)
            if not values:
                minhashes[name][col] = ColumnSignature(None, numeric)
                continue
            # Numeric columns: MinHash over raw values only (equi-join, no
            # embedding/SimHash). Text columns: SBERT -> SimHash -> MinHash
            # (semantic join) when the model is available.
            if backend == "sbert" and not numeric:
                embeddings = model.encode(
                    values, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
                )
                if hyperplanes is None:
                    hyperplanes = _hyperplanes(simhash_size, embeddings.shape[1])
                signature = extract_simhash(embeddings, hyperplanes)
            else:
                signature = values
            minhashes[name][col] = ColumnSignature(
                compute_minhash(signature, num_perm), numeric
            )
    return minhashes


def compute_column_minhashes(
    tables: Dict[str, pd.DataFrame],
    *,
    simhash_size: int = 32,
    num_perm: int = 128,
) -> Dict[str, Dict[str, ColumnSignature]]:
    """One ColumnSignature per column for every table, cached (memory + disk)
    by a fingerprint of the data and parameters."""
    global _model_failed

    def _lookup(backend: str):
        key = _fingerprint(tables, simhash_size, num_perm, backend)
        if key in _MEMO:
            return key, _MEMO[key]
        disk = _CACHE_DIR / f"{key}.pkl"
        if disk.exists():
            try:
                with disk.open("rb") as fh:
                    cached = pickle.load(fh)
                _MEMO[key] = cached
                return key, cached
            except Exception:  # pragma: no cover - corrupt cache
                disk.unlink(missing_ok=True)
        return key, None

    backend = _intended_backend()
    key, hit = _lookup(backend)
    if hit is not None:
        return hit

    # Cache miss: now (and only now) try to load the model. If it is
    # unavailable, fall back to the raw backend and use its cache key.
    if backend == "sbert":
        try:
            _get_model()
        except Exception as exc:  # pragma: no cover - environment dependent
            _model_failed = True
            logger.warning("SBERT unavailable (%s); falling back to raw MinHash.", exc)
            backend = "raw"
            key, hit = _lookup(backend)
            if hit is not None:
                return hit

    disk = _CACHE_DIR / f"{key}.pkl"
    minhashes = _build(tables, simhash_size, num_perm, backend)
    _MEMO[key] = minhashes
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with disk.open("wb") as fh:
            pickle.dump(minhashes, fh)
    except Exception:  # pragma: no cover - best effort
        logger.debug("Could not write minhash cache to %s", disk)
    return minhashes
