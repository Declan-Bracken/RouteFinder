from fastapi import FastAPI

from .api.submit import router as submit_router
from .api.search import router as search_router
from .api.areas import router as areas_router
from .api.feedback import router as feedback_router

app = FastAPI(title="RouteFinder API", version="0.1.0")

app.include_router(submit_router)
app.include_router(search_router)
app.include_router(areas_router)
app.include_router(feedback_router)


@app.get("/health")
def health():
    return {"status": "ok"}
