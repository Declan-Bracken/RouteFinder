from fastapi import Header, HTTPException
from .config import get_settings


def require_admin(x_admin_key: str = Header(default="")):
    key = get_settings().admin_api_key
    if not key:
        raise HTTPException(503, "Admin API key not configured on server")
    if x_admin_key != key:
        raise HTTPException(401, "Invalid or missing X-Admin-Key header")
