"""知识检索工具 - 从向量数据库中检索相关信息"""

import hashlib
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.core.redis_client import get_redis_client
from app.services.vector_store_manager import vector_store_manager


def _cache_key(query: str) -> str:
    return "knowledge:cache:" + hashlib.md5(query.encode()).hexdigest()


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    logger.info(f"知识检索工具被调用: query='{query}'")

    # 查 Redis 缓存
    try:
        redis = get_redis_client()
        cached = redis.get(_cache_key(query))
        if cached is not None:
            logger.info(f"Redis 缓存命中: query='{query}'")
            return cached, []
    except Exception as e:
        logger.warning(f"Redis 不可用，降级走 Milvus: {e}")

    # 缓存未命中，查 Milvus
    try:
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": config.rag_top_k}
        )
        docs = retriever.invoke(query)

        if not docs:
            logger.warning("未检索到相关文档")
            # 空结果也缓存，防止缓存穿透
            try:
                redis.set(_cache_key(query), "", ex=config.redis_cache_empty_ttl)
            except Exception:
                pass
            return "没有找到相关信息。", []

        context = format_docs(docs)
        logger.info(f"检索到 {len(docs)} 个相关文档，写入 Redis 缓存")

        try:
            redis.set(_cache_key(query), context, ex=config.redis_cache_ttl)
        except Exception:
            pass

        return context, docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本
    
    Args:
        docs: 文档列表
        
    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")
        
        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        
        header_str = " > ".join(headers) if headers else ""
        
        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"
        
        formatted_parts.append(formatted)
    
    return "\n".join(formatted_parts)
