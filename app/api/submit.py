import io
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from ..db import get_conn
from ..storage import upload

router = APIRouter()

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_DIMENSION = 1920
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/images/submit")
def submit_image(
    file: UploadFile = File(...),
    route_id: int = Form(...),
    submitted_by: str = Form(default="anonymous"),
    source: str = Form(default="user"),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported type: {file.content_type}")

    data = file.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "Image exceeds 10 MB")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()

    image_id = str(uuid.uuid4())
    b2_key = f"images/routes/{route_id}/{image_id}.jpg"
    upload(b2_key, jpeg_bytes)

    # Admins skip the review queue
    status = "approved" if source == "admin" else "unreviewed"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO images (id, route_id, source, b2_key, submitted_by, status)
                VALUES (%s, %s, %s::image_source, %s, %s, %s::review_status)
                """,
                (image_id, route_id, source, b2_key, submitted_by, status),
            )

    return {"id": image_id, "route_id": route_id, "status": status}
