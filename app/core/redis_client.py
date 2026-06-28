import logging
import redis

from app.config import config

logger = logging.getLogger(__name__)

_pool: redis.ConnectionPool | None = None


def get_redis_client() -> redis.Redis:
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            decode_responses=True,  # 让 get() 直接返回 str，不用手动 .decode()
            max_connections=10,
        )
        logger.info(f"Redis 连接池已创建：{config.redis_host}:{config.redis_port}/db{config.redis_db}")
    return redis.Redis(connection_pool=_pool)
