from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..db import get_conn
from ..storage import presigned_url

router = APIRouter(prefix="/admin/review")


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
                    a.name AS area_name
                FROM images i
                JOIN routes r ON r.id = i.route_id
                JOIN areas  a ON a.id = r.area_id
                WHERE i.status = 'unreviewed'
                ORDER BY i.created_at ASC
                """
            )
            rows = cur.fetchall()

    return {
        "count": len(rows),
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


class ReviewAction(BaseModel):
    action: str               # "approve" or "reject"
    correct_route_id: Optional[int] = None


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
