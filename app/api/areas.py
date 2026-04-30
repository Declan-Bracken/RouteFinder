from fastapi import APIRouter, HTTPException, Query

from ..db import get_conn

router = APIRouter()


def _fmt(slug: str) -> str:
    return slug.replace("-", " ").title() if slug else ""


@router.get("/search")
def unified_search(q: str = Query(..., min_length=2)):
    q_slug = q.strip().replace(" ", "-")
    pattern = f"%{q_slug}%"

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Areas: find matches at any level, count all routes beneath them
            cur.execute(
                """
                WITH RECURSIVE matching AS (
                    SELECT id, name, parent_id FROM areas WHERE name ILIKE %s
                ),
                all_descendants AS (
                    SELECT id, id AS root_id, 0 AS depth FROM matching
                    UNION ALL
                    SELECT a.id, d.root_id, d.depth + 1
                    FROM areas a
                    JOIN all_descendants d ON a.parent_id = d.id
                    WHERE d.depth < 8
                )
                SELECT
                    m.id, m.name, p.name AS parent_name,
                    COUNT(DISTINCT r.id) AS route_count
                FROM matching m
                LEFT JOIN areas p ON p.id = m.parent_id
                LEFT JOIN all_descendants ad ON ad.root_id = m.id
                LEFT JOIN routes r ON r.area_id = ad.id
                GROUP BY m.id, m.name, p.name
                HAVING COUNT(DISTINCT r.id) > 0
                ORDER BY COUNT(DISTINCT r.id) DESC
                LIMIT 8
                """,
                (pattern,),
            )
            area_rows = cur.fetchall()

            # Routes: direct name match with area context
            cur.execute(
                """
                SELECT r.id, r.name, r.grade, a.name AS area_name, p.name AS parent_name
                FROM routes r
                JOIN areas a ON a.id = r.area_id
                LEFT JOIN areas p ON p.id = a.parent_id
                WHERE r.name ILIKE %s
                ORDER BY r.name
                LIMIT 10
                """,
                (pattern,),
            )
            route_rows = cur.fetchall()

    areas = [
        {
            "id": row[0],
            "name": _fmt(row[1]),
            "full_path": f"{_fmt(row[2])} / {_fmt(row[1])}" if row[2] else _fmt(row[1]),
            "route_count": row[3],
        }
        for row in area_rows
    ]

    routes = [
        {
            "id": row[0],
            "name": row[1],
            "grade": row[2],
            "area": f"{_fmt(row[4])} / {_fmt(row[3])}" if row[4] else _fmt(row[3]),
        }
        for row in route_rows
    ]

    return {"areas": areas, "routes": routes}


@router.get("/areas/{area_id}/routes")
def get_routes(area_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM areas WHERE id = %s", (area_id,))
            if not cur.fetchone():
                raise HTTPException(404, f"Area {area_id} not found")

            # Recursively collect all routes under this area at any depth
            cur.execute(
                """
                WITH RECURSIVE sub_areas AS (
                    SELECT id, 0 AS depth FROM areas WHERE id = %s
                    UNION ALL
                    SELECT a.id, s.depth + 1
                    FROM areas a
                    JOIN sub_areas s ON a.parent_id = s.id
                    WHERE s.depth < 8
                )
                SELECT r.id, r.name, r.grade, r.url, a.name AS area_name
                FROM routes r
                JOIN areas a ON a.id = r.area_id
                WHERE r.area_id IN (SELECT id FROM sub_areas)
                ORDER BY a.name, r.name
                """,
                (area_id,),
            )
            rows = cur.fetchall()

    return [
        {"id": row[0], "name": row[1], "grade": row[2], "url": row[3], "area": _fmt(row[4])}
        for row in rows
    ]
