from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..db import get_conn

router = APIRouter()


class FeedbackRequest(BaseModel):
    image_id: str           # UUID of the image that was searched
    confirmed: bool         # True = top result was correct
    correct_route_id: int   # required if confirmed=False


@router.post("/feedback")
def submit_feedback(body: FeedbackRequest):
    if not body.confirmed and body.correct_route_id is None:
        raise HTTPException(400, "correct_route_id is required when confirmed=False")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, route_id FROM images WHERE id = %s", (body.image_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, f"Image {body.image_id} not found")

            if body.confirmed:
                # User confirmed the result was correct — approve as-is
                cur.execute(
                    """
                    UPDATE images
                    SET status = 'approved'::review_status
                    WHERE id = %s
                    """,
                    (body.image_id,),
                )
            else:
                # User provided the correct route — update and approve
                cur.execute(
                    """
                    UPDATE images
                    SET route_id = %s,
                        status   = 'approved'::review_status
                    WHERE id = %s
                    """,
                    (body.correct_route_id, body.image_id),
                )

    return {"status": "ok", "image_id": body.image_id, "confirmed": body.confirmed}
