from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Postgres (Railway provides DATABASE_URL automatically)
    database_url: str

    # Backblaze B2 (S3-compatible)
    b2_key_id: str
    b2_application_key: str
    b2_bucket_name: str
    b2_endpoint_url: str        # e.g. https://s3.us-west-004.backblazeb2.com

    # Model — local path written after download; b2 key is the source of truth
    model_checkpoint: str = ""
    model_checkpoint_b2_key: str = ""
    model_version: str = "v1"

    # Admin API key — set ADMIN_API_KEY in Railway env + mobile .env.local
    admin_api_key: str = ""

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
