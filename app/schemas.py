# -*- coding: utf-8 -*-
"""接口出入参模型。"""
from typing import Optional

from pydantic import BaseModel, Field

from typing import Literal


# 增加crewai的结构化输出模型
class IntentResult(BaseModel):
    intent: Literal[
        "knowledge",
        "order",
        "after_sale_rule",
        "chat",
    ]
    reason: str = Field(description="判断该意图的原因")


class CrewAnswer(BaseModel):
    reply: str
    sources: list[str] = []


class CrewResult(BaseModel):
    reply: str
    intent: str
    reason: str = ""
    sources: list[str] = []

#描述最终HTTP响应的模型
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    reply: str
    intent: Optional[str] = None
    sources: list = []
    engine: str = "langchain"   # langchain | crew
    used_crew: bool = False
    cache_hit: bool = False


class IngestResponse(BaseModel):
    file_name: str
    chunks: int
    total_chunks: int


class StatsResponse(BaseModel):
    total_chunks: int
    llm_model: str
    embedding_model: str
    use_crew: bool
    crew_available: bool
    hybrid_enabled: bool = True
    bm25_ready: bool = False
    rerank_enabled: bool = False
    cache_enabled: bool = False
    cache_size: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_threshold: float = 0.0
    ratelimit_enabled: bool = False
    ratelimit_per_minute: int = 0
    ratelimit_blocked: int = 0
