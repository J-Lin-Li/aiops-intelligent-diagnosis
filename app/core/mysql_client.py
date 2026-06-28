import logging
import aiomysql

from app.config import config

logger = logging.getLogger(__name__)

_pool: aiomysql.Pool | None = None


async def create_pool() -> None:
    global _pool
    _pool = await aiomysql.create_pool(
        host=config.mysql_host,
        port=config.mysql_port,
        user=config.mysql_user,
        password=config.mysql_password,
        db=config.mysql_database,
        autocommit=False,   # 手动控制事务，不自动提交
        minsize=1,
        maxsize=10,
        charset="utf8mb4",
    )
    logger.info(f"MySQL 连接池已创建：{config.mysql_host}:{config.mysql_port}/{config.mysql_database}")


async def close_pool() -> None:
    global _pool
    if _pool:
        _pool.close()
        await _pool.wait_closed()
        logger.info("MySQL 连接池已关闭")


def get_pool() -> aiomysql.Pool:
    if _pool is None:
        raise RuntimeError("MySQL 连接池未初始化，请先调用 create_pool()")
    return _pool
