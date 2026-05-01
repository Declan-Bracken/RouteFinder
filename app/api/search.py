import io

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image

from ..db import get_conn
from ..model import embed_image
from ..config import get_settings

router = APIRouter()


@router.post("/search")
def search(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
):
    data = file.file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # Match submit preprocessing so query and stored embeddings see the same pixels
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
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
            cur.execute(
                """
                SELECT
                    r.id,
                    r.name,
                    r.grade,
                    r.url,
                    a.name          AS area_name,
                    1 - (e.embedding <=> %s::vector) AS similarity
                FROM image_embeddings e
                JOIN images  i ON i.id       = e.image_id
                JOIN routes  r ON r.id       = i.route_id
                JOIN areas   a ON a.id       = r.area_id
                WHERE e.model_version = %s
                  AND i.status = 'approved'
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_str, model_version, vec_str, top_k),
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
