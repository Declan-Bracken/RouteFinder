from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import require_admin
from ..db import get_conn
from ..storage import presigned_url

router = APIRouter(prefix="/admin/review", dependencies=[Depends(require_admin)])


# ── Shared ────────────────────────────────────────────────────────────────────

class ReviewAction(BaseModel):
    action: str               # "approve" or "reject"
    correct_route_id: Optional[int] = None


# ── Images ────────────────────────────────────────────────────────────────────

@router.get("/pending")
def list_pending():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    i.id,
                    i.b2_key,
                    i.submitted_by,
                    i.created_at,
                    r.id   AS route_id,
                    r.name AS route_name,
                    r.grade,
                    a.name AS area_name,
                    (SELECT COUNT(*) FROM images WHERE status = 'unreviewed') AS img_total,
                    (SELECT COUNT(*) FROM areas  WHERE status = 'unreviewed') AS area_total,
                    (SELECT COUNT(*) FROM routes WHERE status = 'unreviewed') AS route_total
                FROM images i
                JOIN routes r ON r.id = i.route_id
                JOIN areas  a ON a.id = r.area_id
                WHERE i.status = 'unreviewed'
                ORDER BY i.created_at ASC
                """
            )
            rows = cur.fetchall()

    img_total   = rows[0][8] if rows else 0
    area_total  = rows[0][9] if rows else 0
    route_total = rows[0][10] if rows else 0

    # If no images, fetch the counts separately
    if not rows:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM images WHERE status = 'unreviewed'),
                        (SELECT COUNT(*) FROM areas  WHERE status = 'unreviewed'),
                        (SELECT COUNT(*) FROM routes WHERE status = 'unreviewed')
                    """
                )
                img_total, area_total, route_total = cur.fetchone()

    return {
        "count":        int(img_total) + int(area_total) + int(route_total),
        "image_count":  int(img_total),
        "area_count":   int(area_total),
        "route_count":  int(route_total),
        "images": [
            {
                "id":           row[0],
                "url":          presigned_url(row[1]),
                "submitted_by": row[2],
                "created_at":   row[3].isoformat() if row[3] else None,
                "route_id":     row[4],
                "route_name":   row[5],
                "grade":        row[6],
                "area":         row[7],
            }
            for row in rows
        ],
    }


@router.post("/{image_id}")
def review_image(image_id: str, body: ReviewAction):
    if body.action not in ("approve", "reject"):
        raise HTTPException(400, "action must be 'approve' or 'reject'")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM images WHERE id = %s", (image_id,))
            if not cur.fetchone():
                raise HTTPException(404, f"Image {image_id} not found")

            if body.action == "reject":
                cur.execute(
                    "UPDATE images SET status = 'rejected'::review_status WHERE id = %s",
                    (image_id,),
                )
            else:
                if body.correct_route_id:
                    cur.execute(
                        """
                        UPDATE images
                        SET status   = 'approved'::review_status,
                            route_id = %s
                        WHERE id = %s
                        """,
                        (body.correct_route_id, image_id),
                    )
                else:
                    cur.execute(
                        "UPDATE images SET status = 'approved'::review_status WHERE id = %s",
                        (image_id,),
                    )

    return {"status": "ok", "image_id": image_id, "action": body.action}


# ── Areas ─────────────────────────────────────────────────────────────────────

@router.get("/pending/areas")
def list_pending_areas():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.id, a.name, a.parent_id, p.name AS parent_name,
                       a.submitted_by, a.created_at
                FROM areas a
                LEFT JOIN areas p ON p.id = a.parent_id
                WHERE a.status = 'unreviewed'
                ORDER BY a.created_at ASC
                """
            )
            rows = cur.fetchall()

    return {
        "count": len(rows),
        "areas": [
            {
                "id":           row[0],
                "name":         row[1],
                "parent_id":    row[2],
                "parent_name":  row[3],
                "submitted_by": row[4],
                "created_at":   row[5].isoformat() if row[5] else None,
            }
            for row in rows
        ],
    }


@router.post("/areas/{area_id}")
def review_area(area_id: int, body: ReviewAction):
    if body.action not in ("approve", "reject"):
        raise HTTPException(400, "action must be 'approve' or 'reject'")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM areas WHERE id = %s AND status = 'unreviewed'",
                (area_id,),
            )
            if not cur.fetchone():
                raise HTTPException(404, f"Pending area {area_id} not found")
            new_status = "approved" if body.action == "approve" else "rejected"
            cur.execute(
                "UPDATE areas SET status = %s::review_status WHERE id = %s",
                (new_status, area_id),
            )
    return {"status": "ok", "area_id": area_id, "action": body.action}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/pending/routes")
def list_pending_routes():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id, r.name, r.grade, r.area_id, a.name AS area_name,
                       r.submitted_by, r.created_at
                FROM routes r
                JOIN areas a ON a.id = r.area_id
                WHERE r.status = 'unreviewed'
                ORDER BY r.created_at ASC
                """
            )
            rows = cur.fetchall()

    return {
        "count": len(rows),
        "routes": [
            {
                "id":           row[0],
                "name":         row[1],
                "grade":        row[2],
                "area_id":      row[3],
                "area":         row[4],
                "submitted_by": row[5],
                "created_at":   row[6].isoformat() if row[6] else None,
            }
            for row in rows
        ],
    }


@router.post("/routes/{route_id}")
def review_route(route_id: int, body: ReviewAction):
    if body.action not in ("approve", "reject"):
        raise HTTPException(400, "action must be 'approve' or 'reject'")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM routes WHERE id = %s AND status = 'unreviewed'",
                (route_id,),
            )
            if not cur.fetchone():
                raise HTTPException(404, f"Pending route {route_id} not found")
            new_status = "approved" if body.action == "approve" else "rejected"
            cur.execute(
                "UPDATE routes SET status = %s::review_status WHERE id = %s",
                (new_status, route_id),
            )
    return {"status": "ok", "route_id": route_id, "action": body.action}
