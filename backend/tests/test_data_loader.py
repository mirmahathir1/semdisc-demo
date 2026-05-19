from __future__ import annotations

import io
import zipfile

import pandas as pd

from semdisc.data_loader import bundled_datalake, drop_fully_blank_rows, list_bundled_datalakes, load_datalake_from_zip


def test_bundled_datalake_loads_csv_tables():
    tables = bundled_datalake("orders")

    assert {"customers", "orders", "order_items", "products"}.issubset(tables)
    assert "customer_id" in tables["customers"].columns


def test_lists_bundled_datalake_folders():
    assert "orders" in list_bundled_datalakes()


def test_drop_fully_blank_rows_removes_whitespace_only_rows():
    df = pd.DataFrame([{"a": "x", "b": "1"}, {"a": "  ", "b": ""}])

    cleaned = drop_fully_blank_rows(df)

    assert cleaned.shape[0] == 1
    assert cleaned.iloc[0]["a"] == "x"


def test_load_datalake_from_zip_reads_csv_members():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("nested/customers.csv", "customer_id,name\nC001,Ada\n")
        zf.writestr("notes.txt", "ignored")

    tables = load_datalake_from_zip(buf.getvalue())

    assert list(tables) == ["customers"]
    assert tables["customers"].iloc[0]["customer_id"] == "C001"
