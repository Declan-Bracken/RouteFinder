import io
import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from PIL import Image, ImageOps

from ..auth import require_admin
from ..db import get_conn
from ..model import embed_image
from ..storage import download

router = APIRouter(prefix="/admin/embed", dependencies=[Depends(require_admin)])
logger = logging.getLogger(__name__)

_job_running = False


def _pending_count(model_version: str) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM images i
                WHERE i.status = 'approved'
                  AND NOT EXISTS (
                      SELECT 1 FROM image_embeddings e
                      WHERE e.image_id = i.id AND e.model_version = %s
                  )
                """,
                (model_version,),
            )
            return cur.fetchone()[0]


def _run_batch(model_version: str):
    global _job_running
    _job_running = True
    embedded = 0
    failed = 0
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT i.id, i.b2_key FROM images i
                    WHERE i.status = 'approved'
                      AND NOT EXISTS (
                          SELECT 1 FROM image_embeddings e
                          WHERE e.image_id = i.id AND e.model_version = %s
                      )
                    """,
                    (model_version,),
                )
                pending = [{"id": str(r[0]), "b2_key": r[1]} for r in cur.fetchall()]

        logger.info(f"Embed job: {len(pending)} images to process (model_version={model_version})")

        for item in pending:
            try:
                data = download(item["b2_key"])
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
                embedding = embed_image(img)
                vec_str = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO image_embeddings (image_id, model_version, embedding)
                            VALUES (%s, %s, %s::vector)
                            ON CONFLICT (image_id, model_version) DO UPDATE
                                SET embedding = EXCLUDED.embedding, computed_at = now()
                            """,
                            (item["id"], model_version, vec_str),
                        )
                embedded += 1
            except Exception as e:
                logger.error(f"Embed job: failed {item['id']}: {e}")
                failed += 1

        logger.info(f"Embed job complete: {embedded} embedded, {failed} failed")
    finally:
        _job_running = False


@router.get("/status")
def embed_status(model_version: str = "v1"):
    return {"pending": _pending_count(model_version), "running": _job_running}


@router.post("")
def trigger_embed(background_tasks: BackgroundTasks, model_version: str = "v1"):
    if _job_running:
        return {"status": "already_running"}
    pending = _pending_count(model_version)
    if pending == 0:
        return {"status": "nothing_to_do", "pending": 0}
    background_tasks.add_task(_run_batch, model_version)
    return {"status": "started", "pending": pending}
