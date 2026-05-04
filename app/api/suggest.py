from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..db import get_conn

router = APIRouter(prefix="/suggest")


class AreaSuggestion(BaseModel):
    name: str
    parent_id: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    submitted_by: Optional[str] = None


class RouteSuggestion(BaseModel):
    name: str
    grade: str
    area_id: int
    type: Optional[str] = None
    submitted_by: Optional[str] = None


@router.post("/area")
def suggest_area(body: AreaSuggestion):
    if not body.name.strip():
        raise HTTPException(400, "Area name is required")
    with get_conn() as conn:
        with conn.cursor() as cur:
            if body.parent_id is not None:
                cur.execute(
                    "SELECT id FROM areas WHERE id = %s AND status = 'approved'",
                    (body.parent_id,),
                )
                if not cur.fetchone():
                    raise HTTPException(400, "Parent area not found or not yet approved")
            cur.execute(
                """
                INSERT INTO areas (name, parent_id, lat, lon, status, submitted_by)
                VALUES (%s, %s, %s, %s, 'unreviewed', %s)
                RETURNING id
                """,
                (body.name.strip(), body.parent_id, body.lat, body.lon, body.submitted_by),
            )
            area_id = cur.fetchone()[0]
    return {"id": area_id, "status": "pending_review"}


@router.post("/route")
def suggest_route(body: RouteSuggestion):
    if not body.name.strip():
        raise HTTPException(400, "Route name is required")
    if not body.grade.strip():
        raise HTTPException(400, "Grade is required")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM areas WHERE id = %s AND status = 'approved'",
                (body.area_id,),
            )
            if not cur.fetchone():
                raise HTTPException(400, "Area not found or not yet approved")
            cur.execute(
                """
                INSERT INTO routes (name, grade, area_id, type, status, submitted_by)
                VALUES (%s, %s, %s, %s, 'unreviewed', %s)
                RETURNING id
                """,
                (body.name.strip(), body.grade.strip(), body.area_id, body.type, body.submitted_by),
            )
            route_id = cur.fetchone()[0]
    return {"id": route_id, "status": "pending_review"}
