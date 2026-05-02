import io
import boto3
from botocore.config import Config as BotoConfig
from functools import lru_cache


@lru_cache
def _client():
    from .config import get_settings
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.b2_endpoint_url,
        aws_access_key_id=s.b2_key_id,
        aws_secret_access_key=s.b2_application_key,
        config=BotoConfig(signature_version="s3v4"),
    )


def upload(key: str, data: bytes, content_type: str = "image/jpeg") -> str:
    from .config import get_settings
    _client().put_object(
        Bucket=get_settings().b2_bucket_name,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    return key


def public_url(key: str) -> str:
    from .config import get_settings
    s = get_settings()
    return f"{s.b2_endpoint_url}/{s.b2_bucket_name}/{key}"


def download(key: str) -> bytes:
    from .config import get_settings
    buf = io.BytesIO()
    _client().download_fileobj(get_settings().b2_bucket_name, key, buf)
    buf.seek(0)
    return buf.read()


def presigned_url(key: str, expires: int = 3600) -> str:
    from .config import get_settings
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": get_settings().b2_bucket_name, "Key": key},
        ExpiresIn=expires,
    )
