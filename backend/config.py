from __future__ import annotations

import os


class Config:
    JSON_SORT_KEYS = False
    MAX_CONTENT_LENGTH = int(os.getenv("SEMDISC_MAX_CONTENT_LENGTH", str(50 * 1024 * 1024)))

