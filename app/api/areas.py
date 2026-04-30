from fastapi import APIRouter, HTTPException, Query

from ..db import get_conn

router = APIRouter()


def _fmt(slug: str) -> str:
    return slug.replace("-", " ").title() if slug else ""


@router.get("/areas/search")
def search_areas(q: str = Query(..., min_length=2)):
    q_slug = q.strip().replace(" ", "-")
    pattern = f"%{q_slug}%"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH RECURSIVE matching AS (
                    SELECT id FROM areas WHERE name ILIKE %s
                ),
                descendants AS (
                    SELECT id, 0 AS depth FROM matching
                    UNION ALL
                    SELECT a.id, d.depth + 1
                    FROM areas a
                    JOIN descendants d ON a.parent_id = d.id
                    WHERE d.depth < 8
                )
                SELECT DISTINCT
                    a.id, a.name, a.url, p.name AS parent_name,
                    CASE WHEN a.name ILIKE %s THEN 0 ELSE 1 END AS rank
                FROM areas a
                LEFT JOIN areas p ON p.id = a.parent_id
                WHERE a.id IN (SELECT id FROM descendants)
                  AND EXISTS (SELECT 1 FROM routes r WHERE r.area_id = a.id)
                ORDER BY rank, a.name
                LIMIT 20
                """,
                (pattern, pattern),
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "name": _fmt(row[1]),
            "url": row[2],
            "full_path": f"{_fmt(row[3])} / {_fmt(row[1])}" if row[3] else _fmt(row[1]),
        }
        for row in rows
    ]


@router.get("/areas/{area_id}/routes")
def get_routes(area_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM areas WHERE id = %s", (area_id,))
            if not cur.fetchone():
                raise HTTPException(404, f"Area {area_id} not found")

            cur.execute(
                """
                SELECT id, name, grade, url
                FROM routes
                WHERE area_id = %s
                ORDER BY name
                """,
                (area_id,),
            )
            rows = cur.fetchall()

    return [
        {"id": row[0], "name": row[1], "grade": row[2], "url": row[3]}
        for row in rows
    ]
