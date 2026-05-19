from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EdgeInfo:
    left: str
    right: str
    key: str
    score: float
    right_key: Optional[str] = None
    join_type: str = "equi"

