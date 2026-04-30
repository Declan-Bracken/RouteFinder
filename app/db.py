from contextlib import contextmanager
from psycopg2.pool import ThreadedConnectionPool

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        from .config import get_settings
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=get_settings().database_url)
    return _pool


@contextmanager
def get_conn():
    """Yields a psycopg2 connection from the pool. Commits on success, rolls back on error."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
