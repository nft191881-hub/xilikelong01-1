import os
from typing import List, Optional

import psycopg2
from psycopg2.extras import Json

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inference_results (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_type   TEXT NOT NULL,
    source_url    TEXT,
    image_name    TEXT,
    frontend      TEXT,
    backend       TEXT,
    database_name TEXT,
    confidence    DOUBLE PRECISION,
    evidence      JSONB,
    skeleton      JSONB,
    raw_text      TEXT
);
CREATE INDEX IF NOT EXISTS idx_inference_results_created_at
    ON inference_results (created_at DESC);
"""


def get_connection():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        return None
    return psycopg2.connect(dsn, connect_timeout=5)


def init_db() -> dict:
    """Create the table if it does not exist. Safe to call on every startup."""
    conn = get_connection()
    if conn is None:
        return {"initialized": False, "reason": "DATABASE_URL is not set"}
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        return {"initialized": True}
    except Exception as exc:
        conn.rollback()
        return {"initialized": False, "error": str(exc)}
    finally:
        conn.close()


def save_result(
    source_type: str,
    source_url: Optional[str],
    image_name: Optional[str],
    frontend: Optional[str],
    backend: Optional[str],
    database_name: Optional[str],
    confidence: float,
    evidence: List[str],
    skeleton: List[str],
    raw_text: str,
) -> dict:
    try:
        conn = get_connection()
    except Exception as exc:
        return {"stored": False, "error": str(exc)}
    if conn is None:
        return {"stored": False, "reason": "DATABASE_URL is not set"}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO inference_results (
                    source_type, source_url, image_name,
                    frontend, backend, database_name,
                    confidence, evidence, skeleton, raw_text
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    source_type,
                    source_url,
                    image_name,
                    frontend,
                    backend,
                    database_name,
                    confidence,
                    Json(evidence),
                    Json(skeleton),
                    (raw_text or "")[:5000],
                ),
            )
        conn.commit()
        return {"stored": True}
    except Exception as exc:
        conn.rollback()
        return {"stored": False, "error": str(exc)}
    finally:
        conn.close()


def get_recent_results(limit: int = 10) -> list:
    limit = max(1, min(int(limit), 100))
    try:
        conn = get_connection()
    except Exception:
        return []
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, source_type, source_url, image_name,
                       frontend, backend, database_name, confidence
                FROM inference_results
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "created_at": r[1].isoformat() if r[1] else None,
                "source_type": r[2],
                "source_url": r[3],
                "image_name": r[4],
                "frontend": r[5],
                "backend": r[6],
                "database_name": r[7],
                "confidence": float(r[8]) if r[8] is not None else None,
            }
            for r in rows
        ]
    except Exception:
        return []
    finally:
        conn.close()
