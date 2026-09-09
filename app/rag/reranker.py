# -*- coding: utf-8 -*-
"""bge-reranker 重排：支持本地 CrossEncoder 与在线硅基流动 API 两种 provider。

- local        ：sentence_transformers.CrossEncoder 本地加载（懒加载 + HF 镜像 + 异常降级）
- siliconflow  ：调用 POST {base_url}/rerank（零下载、零本地 CPU），异常降级为融合顺序

无论哪种 provider，加载/调用失败都返回原顺序，不影响主链路。
"""
import logging
import os
from typing import List

from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger("airobot.reranker")

# 硅基流动在线重排默认模型（免费档）；本地 provider 使用 settings.rerank_model 指定的 bge 模型
SILICONFLOW_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
_LOCAL_DEFAULT_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self) -> None:
        self._model = None

    @property
    def ready(self) -> bool:
        if not settings.rerank_enabled:
            return False
        # 在线 provider：只校验是否有 Key，不加载任何本地模型
        if settings.rerank_provider == "siliconflow":
            return bool(settings.rerank_api_key)
        if self._model is not None:
            return True
        try:
            # 国内网络默认走 HF 镜像，避免 huggingface.co 连接超时挂起
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from sentence_transformers import CrossEncoder
            try:
                # 优先本地缓存加载（离线可用）；未缓存时走镜像下载
                self._model = CrossEncoder(
                    settings.rerank_model, max_length=512, local_files_only=True)
            except Exception:
                self._model = CrossEncoder(settings.rerank_model, max_length=512)
            return True
        except Exception as exc:
            logger.warning("重排模型加载失败，自动跳过重排: %s", exc)
            return False

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """打分重排，返回 top_k；任何异常都回退为原顺序。"""
        if not docs:
            return docs
        if not self.ready:
            return docs[:top_k]
        if settings.rerank_provider == "siliconflow":
            return self._rerank_siliconflow(query, docs, top_k)
        try:
            pairs = [(query, d.page_content) for d in docs]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        except Exception as exc:
            logger.warning("重排打分失败，保留融合顺序: %s", exc)
            return docs[:top_k]
        return [d for d, _ in ranked[:top_k]]

    def _rerank_siliconflow(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """在线重排：POST {base_url}/rerank，按 relevance_score 降序返回（index 指向原文档）。"""
        import httpx

        model = settings.rerank_model
        # 未针对 API 单独配置模型时，自动切换为硅基流动默认重排模型
        if model == _LOCAL_DEFAULT_MODEL:
            model = SILICONFLOW_RERANK_MODEL
        try:
            resp = httpx.post(
                f"{settings.rerank_base_url.rstrip('/')}/rerank",
                headers={"Authorization": f"Bearer {settings.rerank_api_key}",
                         "Content-Type": "application/json"},
                json={"model": model, "query": query,
                      "documents": [d.page_content for d in docs], "top_n": top_k},
                timeout=30.0,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            order = [r["index"] for r in results if r.get("index") is not None]
            ranked = [docs[i] for i in order if 0 <= i < len(docs)]
            # API 返回条数不足时，用原顺序补齐，保证返回 top_k 条
            seen = set(order)
            ranked += [d for i, d in enumerate(docs) if i not in seen]
            return ranked[:top_k]
        except Exception as exc:
            logger.warning("硅基流动重排调用失败，保留融合顺序: %s", exc)
            return docs[:top_k]
