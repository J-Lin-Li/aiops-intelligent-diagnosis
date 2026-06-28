"""向量嵌入服务模块 - 基于 LangChain Embeddings 标准接口"""

from typing import List

import dashscope
from langchain_core.embeddings import Embeddings
from loguru import logger

from app.config import config


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope 多模态 Embedding（原生接口）

    实现 LangChain 标准 Embeddings 接口:
    - embed_documents(texts: List[str]) → List[List[float]]: 批量嵌入文档
    - embed_query(text: str) → List[float]: 嵌入单个查询
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
    ):
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions

        masked_key = self._mask_api_key(api_key)
        logger.info(
            f"DashScope Embeddings 初始化完成 - "
            f"模型: {model}, 维度: {dimensions}, API Key: {masked_key}"
        )

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        if len(api_key) > 8:
            return f"{api_key[:8]}...{api_key[-4:]}"
        return "***"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            logger.info(f"批量嵌入 {len(texts)} 个文档")
            response = dashscope.MultiModalEmbedding.call(
                model=self.model,
                input=[{"text": t} for t in texts],
                api_key=self.api_key,
            )
            if response.status_code != 200:
                raise RuntimeError(f"DashScope API 错误: {response.message}")
            embeddings = [item['embedding'] for item in response.output['embeddings']]
            logger.debug(f"批量嵌入完成, 维度: {len(embeddings[0])}")
            return embeddings
        except Exception as e:
            logger.error(f"批量嵌入失败: {e}")
            raise RuntimeError(f"批量嵌入失败: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")
        try:
            logger.debug(f"嵌入查询, 长度: {len(text)} 字符")
            response = dashscope.MultiModalEmbedding.call(
                model=self.model,
                input=[{"text": text}],
                api_key=self.api_key,
            )
            if response.status_code != 200:
                raise RuntimeError(f"DashScope API 错误: {response.message}")
            embedding = response.output['embeddings'][0]['embedding']
            logger.debug(f"查询嵌入完成, 维度: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise RuntimeError(f"查询嵌入失败: {e}") from e


# 全局单例
vector_embedding_service = DashScopeEmbeddings(
    api_key=config.dashscope_api_key,
    model=config.dashscope_embedding_model,
    dimensions=config.dashscope_embedding_dim,
)
