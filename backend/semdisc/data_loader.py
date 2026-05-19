from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATALAKE_DIR = PROJECT_ROOT / "datalake"
DEFAULT_DATALAKE = "orders"


def read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


def list_bundled_datalakes() -> list[str]:
    names = [
        path.name
        for path in DATALAKE_DIR.iterdir()
        if path.is_dir() and any(child.suffix.lower() == ".csv" for child in path.iterdir())
    ]
    return sorted(names)


def _datalake_path(name: str | None = None) -> Path:
    names = list_bundled_datalakes()
    selected = name or (DEFAULT_DATALAKE if DEFAULT_DATALAKE in names else names[0] if names else "")
    path = DATALAKE_DIR / selected
    if selected and path.is_dir():
        return path
    return DATALAKE_DIR


@lru_cache(maxsize=16)
def _bundled_datalake_cached(name: str | None = None) -> tuple[tuple[str, pd.DataFrame], ...]:
    tables: Dict[str, pd.DataFrame] = {}
    for path in sorted(_datalake_path(name).glob("*.csv")):
        if path.stem == "query":
            continue
        df = pd.read_csv(path)
        if df.empty or df.shape[1] == 0:
            continue
        tables[path.stem] = df
    return tuple((name, df.copy()) for name, df in tables.items())


def bundled_datalake(name: str | None = None) -> Dict[str, pd.DataFrame]:
    return {name: df.copy() for name, df in _bundled_datalake_cached(name)}


@lru_cache(maxsize=16)
def _bundled_query_cached(name: str | None = None) -> pd.DataFrame:
    query_path = _datalake_path(name) / "query.csv"
    if query_path.exists():
        return pd.read_csv(query_path)
    return pd.DataFrame(
        [
            {"order_id": "O100", "customer_id": "C001", "country": "US"},
            {"order_id": "O101", "customer_id": "C002", "country": "SG"},
            {"order_id": "O102", "customer_id": "C001", "country": "US"},
            {"order_id": "O103", "customer_id": "C003", "country": "PL"},
        ]
    )


def bundled_query(name: str | None = None) -> pd.DataFrame:
    return _bundled_query_cached(name).copy()


def load_datalake_from_zip(zip_bytes: bytes) -> Dict[str, pd.DataFrame]:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    tables: Dict[str, pd.DataFrame] = {}
    for info in zf.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".csv"):
            continue
        name = Path(info.filename).stem.strip()
        if not name:
            continue
        df = read_csv_bytes(zf.read(info))
        if df.empty or df.shape[1] == 0:
            continue
        tables[name] = df
    return tables


def load_datalake_from_uploads(files: Iterable[tuple[str, bytes]]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for filename, content in files:
        name = Path(filename).name
        if name.lower().endswith(".zip"):
            tables.update(load_datalake_from_zip(content))
            continue
        if not name.lower().endswith(".csv"):
            continue
        table_name = Path(name).stem.strip()
        if table_name:
            tables[table_name] = read_csv_bytes(content)
    return ensure_nonempty_tables(tables)


def ensure_nonempty_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cleaned = {}
    for name, df in tables.items():
        df2 = drop_fully_blank_rows(df)
        if df2 is None or df2.empty or df2.shape[1] == 0:
            continue
        cleaned[str(name)] = df2
    return cleaned


def drop_fully_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or df.shape[1] == 0:
        return df
    df2 = df.copy()
    obj_cols = df2.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) > 0:
        df2[obj_cols] = df2[obj_cols].astype("string").apply(lambda s: s.str.strip())
        df2[obj_cols] = df2[obj_cols].replace(r"^\s*$", pd.NA, regex=True)
    return df2.dropna(how="all")
