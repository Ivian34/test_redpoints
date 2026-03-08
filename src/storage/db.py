from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analysed_listings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at_utc TEXT NOT NULL,
    title TEXT NOT NULL,
    stage_1_ran INTEGER NOT NULL,
    stage_2_ran INTEGER NOT NULL,
    similarity_ran INTEGER NOT NULL,
    asset_score REAL NOT NULL,
    is_asset INTEGER NOT NULL,
    similarity_score REAL NOT NULL,
    top_k INTEGER NOT NULL,
    top_k_most_similar_reference_listings_json TEXT NOT NULL,
    suspicion_score REAL,
    suspicion_flag INTEGER
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_analysed_listings_asset_score ON analysed_listings(asset_score);",
    "CREATE INDEX IF NOT EXISTS idx_analysed_listings_suspicion_score ON analysed_listings(suspicion_score);",
    "CREATE INDEX IF NOT EXISTS idx_analysed_listings_is_asset ON analysed_listings(is_asset);",
    "CREATE INDEX IF NOT EXISTS idx_analysed_listings_created_at ON analysed_listings(created_at_utc DESC);",
]


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def init_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        for index_sql in CREATE_INDEXES_SQL:
            conn.execute(index_sql)
        conn.commit()


def insert_analysed_listing(db_path: Path, record: dict[str, Any]) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": str(record["title"]),
        "stage_1_ran": int(bool(record["stage_1_ran"])),
        "stage_2_ran": int(bool(record["stage_2_ran"])),
        "similarity_ran": int(bool(record["similarity_ran"])),
        "asset_score": float(record["asset_score"]),
        "is_asset": int(bool(record["is_asset"])),
        "similarity_score": float(record["similarity_score"]),
        "top_k": int(record["top_k"]),
        "top_k_most_similar_reference_listings_json": json.dumps(
            record["top_k_most_similar_reference_listings"], ensure_ascii=False
        ),
        "suspicion_score": (
            float(record["suspicion_score"])
            if record.get("suspicion_score") is not None
            else None
        ),
        "suspicion_flag": (
            int(bool(record["suspicion_flag"]))
            if record.get("suspicion_flag") is not None
            else None
        ),
    }

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO analysed_listings (
                created_at_utc,
                title,
                stage_1_ran,
                stage_2_ran,
                similarity_ran,
                asset_score,
                is_asset,
                similarity_score,
                top_k,
                top_k_most_similar_reference_listings_json,
                suspicion_score,
                suspicion_flag
            ) VALUES (
                :created_at_utc,
                :title,
                :stage_1_ran,
                :stage_2_ran,
                :similarity_ran,
                :asset_score,
                :is_asset,
                :similarity_score,
                :top_k,
                :top_k_most_similar_reference_listings_json,
                :suspicion_score,
                :suspicion_flag
            )
            """,
            payload,
        )
        conn.commit()


def get_analysed_listings_above_threshold(
    db_path: Path, stage: int, threshold: float
) -> list[dict[str, Any]]:
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")

    base_select = """
        SELECT
            id,
            created_at_utc,
            title,
            stage_1_ran,
            stage_2_ran,
            similarity_ran,
            asset_score,
            is_asset,
            similarity_score,
            top_k,
            top_k_most_similar_reference_listings_json,
            suspicion_score,
            suspicion_flag
        FROM analysed_listings
    """

    if stage == 1:
        sql = (
            base_select
            + " WHERE asset_score >= ? ORDER BY created_at_utc DESC, id DESC"
        )
        params = (float(threshold),)
    else:
        sql = (
            base_select
            + " WHERE is_asset = 1 AND suspicion_score IS NOT NULL "
              "AND suspicion_score >= ? ORDER BY created_at_utc DESC, id DESC"
        )
        params = (float(threshold),)

    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        top_k_json = row["top_k_most_similar_reference_listings_json"] or "[]"
        results.append(
            {
                "id": int(row["id"]),
                "created_at_utc": row["created_at_utc"],
                "title": row["title"],
                "stage_1_ran": bool(row["stage_1_ran"]),
                "stage_2_ran": bool(row["stage_2_ran"]),
                "similarity_ran": bool(row["similarity_ran"]),
                "is_asset": bool(row["is_asset"]),
                "asset_score": float(row["asset_score"]),
                "suspicion_score": (
                    float(row["suspicion_score"])
                    if row["suspicion_score"] is not None
                    else None
                ),
                "suspicion_flag": (
                    bool(row["suspicion_flag"])
                    if row["suspicion_flag"] is not None
                    else None
                ),
                "similarity_score": float(row["similarity_score"]),
                "top_k": int(row["top_k"]),
                "top_k_most_similar_reference_listings": json.loads(top_k_json),
            }
        )
    return results

def get_last_n_analysed_listings_db(db_path: Path, n: int) -> list[dict[str, Any]]:
    if n <= 0:
        raise ValueError("n must be > 0")

    base_select = """
        SELECT
            id,
            created_at_utc,
            title,
            stage_1_ran,
            stage_2_ran,
            similarity_ran,
            asset_score,
            is_asset,
            similarity_score,
            top_k,
            top_k_most_similar_reference_listings_json,
            suspicion_score,
            suspicion_flag
        FROM analysed_listings
    """

    sql = base_select + " ORDER BY created_at_utc DESC, id DESC LIMIT ?"
    params = (int(n),)

    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        top_k_json = row["top_k_most_similar_reference_listings_json"] or "[]"
        results.append(
            {
                "id": int(row["id"]),
                "created_at_utc": row["created_at_utc"],
                "title": row["title"],
                "stage_1_ran": bool(row["stage_1_ran"]),
                "stage_2_ran": bool(row["stage_2_ran"]),
                "similarity_ran": bool(row["similarity_ran"]),
                "is_asset": bool(row["is_asset"]),
                "asset_score": float(row["asset_score"]),
                "suspicion_score": (
                    float(row["suspicion_score"])
                    if row["suspicion_score"] is not None
                    else None
                ),
                "suspicion_flag": (
                    bool(row["suspicion_flag"])
                    if row["suspicion_flag"] is not None
                    else None
                ),
                "similarity_score": float(row["similarity_score"]),
                "top_k": int(row["top_k"]),
                "top_k_most_similar_reference_listings": json.loads(top_k_json),
            }
        )
    return results
