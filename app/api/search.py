import io
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image, ImageOps

from ..db import get_conn
from ..model import embed_image
from ..config import get_settings

router = APIRouter()


@router.post("/search")
def search(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
    area_id: Optional[int] = Query(default=None),
):
    data = file.file.read()
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.LANCZOS)
    except Exception:
        raise HTTPException(400, "Could not decode image")

    try:
        embedding = embed_image(img)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    model_version = get_settings().model_version
    vec_str = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

    with get_conn() as conn:
        with conn.cursor() as cur:
            if area_id is not None:
                cur.execute(
                    """
                    WITH RECURSIVE sub_areas AS (
                        SELECT id FROM areas WHERE id = %s
                        UNION ALL
                        SELECT a.id FROM areas a JOIN sub_areas s ON a.parent_id = s.id
                    ),
                    combined AS (
                        SELECT r.id, r.name, r.grade, r.url, a.name AS area_name,
                               1 - (e.embedding <=> %s::vector) AS similarity
                        FROM image_embeddings e
                        JOIN images i ON i.id = e.image_id
                        JOIN routes r ON r.id = i.route_id
                        JOIN areas  a ON a.id = r.area_id
                        WHERE e.model_version = %s AND i.status = 'approved'
                          AND r.area_id IN (SELECT id FROM sub_areas)
                        UNION ALL
                        SELECT r.id, r.name, r.grade, r.url, a.name AS area_name,
                               1 - (e.embedding <=> %s::vector) AS similarity
                        FROM route_image_embeddings e
                        JOIN route_images ri ON ri.id = e.route_image_id
                        JOIN routes       r  ON r.id  = ri.route_id
                        JOIN areas        a  ON a.id  = r.area_id
                        WHERE e.model_version = %s AND r.status = 'approved'
                          AND r.area_id IN (SELECT id FROM sub_areas)
                    )
                    SELECT id, name, grade, url, area_name, MAX(similarity) AS similarity
                    FROM combined
                    GROUP BY id, name, grade, url, area_name
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (area_id, vec_str, model_version, vec_str, model_version, top_k),
                )
            else:
                cur.execute(
                    """
                    WITH combined AS (
                        SELECT r.id, r.name, r.grade, r.url, a.name AS area_name,
                               1 - (e.embedding <=> %s::vector) AS similarity
                        FROM image_embeddings e
                        JOIN images i ON i.id = e.image_id
                        JOIN routes r ON r.id = i.route_id
                        JOIN areas  a ON a.id = r.area_id
                        WHERE e.model_version = %s AND i.status = 'approved'
                        UNION ALL
                        SELECT r.id, r.name, r.grade, r.url, a.name AS area_name,
                               1 - (e.embedding <=> %s::vector) AS similarity
                        FROM route_image_embeddings e
                        JOIN route_images ri ON ri.id = e.route_image_id
                        JOIN routes       r  ON r.id  = ri.route_id
                        JOIN areas        a  ON a.id  = r.area_id
                        WHERE e.model_version = %s AND r.status = 'approved'
                    )
                    SELECT id, name, grade, url, area_name, MAX(similarity) AS similarity
                    FROM combined
                    GROUP BY id, name, grade, url, area_name
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (vec_str, model_version, vec_str, model_version, top_k),
                )
            rows = cur.fetchall()

    return {
        "results": [
            {
                "route_id":   row[0],
                "name":       row[1],
                "grade":      row[2],
                "url":        row[3],
                "area":       row[4],
                "similarity": round(float(row[5]), 4),
            }
            for row in rows
        ]
    }
