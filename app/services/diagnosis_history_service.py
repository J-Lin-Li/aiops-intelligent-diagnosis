import logging
from typing import List, Tuple

from app.core.mysql_client import get_pool

logger = logging.getLogger(__name__)


async def save_diagnosis(
    session_id: str,
    past_steps: List[Tuple[str, str]],
    report: str,
) -> int | None:
    pool = get_pool()
    async with pool.acquire() as conn:  #池里借一个连接，with 块结束后自动归还，
        async with conn.cursor() as cur:    #cursor（游标）：执行 SQL 的工具对象
            try:
                await conn.begin()  # 发命令给 MySQL，等 MySQL 确认

                # 写主记录，发 SQL 给 MySQL，等 MySQL 执行完,%s占位符，存入的内容
                await cur.execute(
                    "INSERT INTO diagnosis_history (session_id, report) VALUES (%s, %s)",
                    (session_id, report),
                )
                #INSERT 执行后，MySQL 自动生成的自增 id 就在这里
                history_id = cur.lastrowid

                # 批量写步骤明细
                if past_steps:
                    step_rows = [
                        (history_id, index, step[0], step[1])
                        for index, step in enumerate(past_steps)
                    ]
                    # 发 SQL 给 MySQL
                    await cur.executemany(
                        "INSERT INTO diagnosis_steps (history_id, step_index, step_desc, step_result) "
                        "VALUES (%s, %s, %s, %s)",
                        step_rows,
                    )

                await conn.commit() # 发 SQL 给 MySQL，提交
                logger.info(f"诊断历史已保存: session_id={session_id}, history_id={history_id}, steps={len(past_steps)}")
                return history_id

            except Exception as e:
                await conn.rollback()   # 发 SQL 给 MySQL，回滚
                logger.error(f"诊断历史保存失败，已回滚: {e}")
                return None
