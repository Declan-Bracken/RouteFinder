from fastapi import APIRouter, HTTPException, Query

from ..db import get_conn

router = APIRouter()


@router.get("/areas/search")
def search_areas(q: str = Query(..., min_length=2)):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH RECURSIVE ancestors AS (
                    SELECT id, name, parent_id, name::text AS path
                    FROM areas
                    UNION ALL
                    SELECT a.id, a.name, a.parent_id, a.name || ' / ' || anc.path
                    FROM areas a
                    JOIN ancestors anc ON a.id = anc.parent_id
                ),
                full_paths AS (
                    SELECT id, MAX(path) AS full_path
                    FROM ancestors
                    GROUP BY id
                )
                SELECT a.id, a.name, a.url, fp.full_path
                FROM areas a
                JOIN full_paths fp ON fp.id = a.id
                WHERE a.name ILIKE %s
                ORDER BY a.name
                LIMIT 20
                """,
                (f"%{q}%",),
            )
            rows = cur.fetchall()

    return [{"id": row[0], "name": row[1], "url": row[2], "full_path": row[3]} for row in rows]


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
