"""
SQLite-based screening history for the Dyslexia Screening System.

Stores every analysis run so users can track a child's progress over time
and export results as a CSV summary.
"""

import csv
import io
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

DB_PATH = Path(config.BASE_DIR) / "screening_history.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS screenings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    filename         TEXT,
    final_risk       REAL,
    risk_label       TEXT,
    handwriting_risk REAL,
    language_risk    REAL,
    reversal_ratio   REAL,
    regularity_risk  REAL,
    total_patches    INTEGER,
    dyslexic_patches INTEGER
)
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def _init() -> None:
    with _conn() as c:
        c.execute(_CREATE_SQL)
        c.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_screening(
    filename: str,
    final_risk: float,
    risk_label: str,
    handwriting_risk: float,
    language_risk: Optional[float],
    reversal_ratio: float = 0.0,
    regularity_risk: float = 0.5,
    total_patches: int = 0,
    dyslexic_patches: int = 0,
) -> None:
    """Persist one screening result to the database."""
    try:
        _init()
        with _conn() as c:
            c.execute(
                """
                INSERT INTO screenings
                  (timestamp, filename, final_risk, risk_label,
                   handwriting_risk, language_risk, reversal_ratio,
                   regularity_risk, total_patches, dyslexic_patches)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    filename or "unknown",
                    round(final_risk, 4),
                    risk_label,
                    round(handwriting_risk, 4),
                    round(language_risk, 4) if language_risk is not None else None,
                    round(reversal_ratio, 4),
                    round(regularity_risk, 4),
                    total_patches,
                    dyslexic_patches,
                ),
            )
            c.commit()
    except Exception as exc:
        logger.warning("Could not save screening history: %s", exc)


def get_history(limit: int = 100) -> List[Dict]:
    """Return the most recent screenings, newest first."""
    try:
        _init()
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM screenings ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Could not load screening history: %s", exc)
        return []


def clear_history() -> None:
    """Delete all screening records."""
    try:
        _init()
        with _conn() as c:
            c.execute("DELETE FROM screenings")
            c.commit()
    except Exception as exc:
        logger.warning("Could not clear history: %s", exc)


def export_csv() -> bytes:
    """Export all history rows as a UTF-8 CSV byte string."""
    rows = get_history(limit=10_000)
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")
