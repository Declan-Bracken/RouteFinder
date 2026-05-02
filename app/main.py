from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.submit import router as submit_router
from .api.search import router as search_router
from .api.areas import router as areas_router
from .api.feedback import router as feedback_router
from .api.review import router as review_router
from .api.embed import router as embed_router

app = FastAPI(title="RouteFinder API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(submit_router)
app.include_router(search_router)
app.include_router(areas_router)
app.include_router(feedback_router)
app.include_router(review_router)
app.include_router(embed_router)


@app.get("/health")
def health():
    return {"status": "ok"}
