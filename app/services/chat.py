# -*- coding: utf-8 -*-
"""对话编排服务（异步版）：
1) 优先 CrewAI 多智能体（未安装/异常时自动降级，多 Agent 在独立线程运行）；
2) 降级路径：LangChain 意图分类路由 + RAG/工具；
3) 所有路径带会话记忆（session_id），支持多轮上下文。
"""
import asyncio
import json
import logging
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.agents.tools import CREW_TOOLS_READY, query_order
from app.config import settings
from app.rag.retriever import aanswer_with_rag, kb
from app.services.resilience import ainvoke_with_retry
from app.services.semantic_cache import semantic_cache
from app.services.memory import format_history, memory
from app.services.tracing import traces

logger = logging.getLogger("airobot.chat")

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是意图分类器，只输出 JSON：{{\"intent\": \"knowledge|order|after_sale_rule|chat\", \"reason\": \"简短理由\"}}"),
    ("human", "{message}"),
])

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是二手交易平台的智能客服，语气友好简洁；涉及订单或平台规则时引导用户使用对应功能。"),
    MessagesPlaceholder("history"),
    ("human", "{message}"),
])

_llm_cache: dict = {}


def get_llm() -> ChatOpenAI:
    key = (settings.llm_model, settings.llm_base_url, settings.llm_api_key)
    if key not in _llm_cache:
        _llm_cache[key] = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key or "sk-placeholder-not-configured",
            base_url=settings.llm_base_url,
            temperature=0.3,
        )
    return _llm_cache[key]


async def classify_intent(message: str) -> str:
    chain = INTENT_PROMPT | get_llm() | StrOutputParser()
    raw = await ainvoke_with_retry(chain.ainvoke, {"message": message})
    try:
        data = json.loads(raw.strip().strip("`"))
        return data.get("intent", "chat")
    except Exception:
        logger.warning("意图 JSON 解析失败，默认 chat: %s", raw[:80])
        return "chat"


async def fallback_chat(message: str, session_id: str) -> dict:
    """内置路由：不依赖 CrewAI，逻辑与 Crew 中 Task 一致；带会话记忆。"""
    history = memory.get_messages(session_id)
    intent = await classify_intent(message)
    if intent == "order":
        return {"reply": query_order(message), "intent": intent, "sources": [], "engine": "langchain"}
    if intent == "knowledge":
        answer, sources = await aanswer_with_rag(message, get_llm(), history)
        return {"reply": answer, "intent": intent, "sources": sources, "engine": "langchain"}
    chain = CHAT_PROMPT | get_llm() | StrOutputParser()
    reply = await ainvoke_with_retry(chain.ainvoke, {"message": message, "history": history})
    return {"reply": reply, "intent": "chat", "sources": [], "engine": "langchain"}


def _maybe_cache(query_vec: list | None, result: dict, message: str = "") -> None:
    """动态数据（订单）不缓存；无上下文问题才写入语义缓存。"""
    if query_vec is not None and result.get("intent") != "order":
        semantic_cache.put(query_vec, result, message)


async def chat(message: str, session_id: str = "default") -> dict:
    """对话入口：CrewAI 优先（线程池执行），失败自动降级内置路由；成功后写入会话记忆。"""
    entry = {"message": message[:80], "session_id": session_id, "status": 200}
    t_start = time.perf_counter()
    if not settings.llm_api_key:
        traces.record({**entry, "intent": "no-key", "total_ms": 0.0})
        return {"reply": "未配置 AIROBOT_LLM_API_KEY，请复制 .env.example 为 .env 并填入密钥。",
                "intent": None, "sources": [], "engine": "langchain"}

    # 语义缓存：仅无上下文的首轮问题参与命中/写入（避免与会话记忆耦合）
    query_vec = None
    if settings.cache_enabled and settings.embedding_api_key and not memory.get_messages(session_id):
        t_cache = time.perf_counter()
        query_vec = await asyncio.to_thread(kb.embed_query, message)
        cached = semantic_cache.get(query_vec, message)
        entry["cache_lookup_ms"] = round((time.perf_counter() - t_cache) * 1000, 1)
        entry["cache_checked"] = True
        if cached is not None:
            entry.update(cache_hit=True, intent=cached.get("intent"),
                         total_ms=round((time.perf_counter() - t_start) * 1000, 1))
            traces.record(entry)
            return {**cached, "cache_hit": True}
    else:
        entry["cache_lookup_ms"] = 0.0

    t_llm = time.perf_counter()
    if settings.use_crew and CREW_TOOLS_READY:
        try:
            #memory.get_messages(session_id) 取出该会话的多轮历史消息
            #format_history() 把消息列表拼成自然语言文本，方便后续直接塞进 Prompt
            #因为 CrewAI 的 Task 期望的是纯字符串输入，而不是 LangChain 的 MessagesPlaceholder
            history_text = format_history(memory.get_messages(session_id))  
            from app.agents.crew import run_crew
            """
            CrewAI 的 kickoff() 是同步阻塞的（内部有循环、sleep、API 调用等），
            而 chat() 是 async def 异步函数。如果直接调用 run_crew()，会卡住整个事件循环，导致其他请求无法响应。
            用 to_thread() 相当于"开个后台线程跑，跑完告诉我结果"。
            """
            ##async def to_thread(func, /, *args, **kwargs),传递给这个函数的参数都会直接传递给 func。# run_crew(message: str, history_text: str = "") -> str:
            crew_result = await asyncio.to_thread(run_crew, message, history_text)   
            # CrewResult 的 sources 可能为空；先规范为列表，避免未定义变量或 None
            # 进入缓存、追踪和 FastAPI 响应转换逻辑。
            sources = [str(item) for item in (crew_result.sources or [])]
            result = {"reply": crew_result.reply, "intent": crew_result.intent, "sources": sources,
                      "engine": "crew", "used_crew": True}
            #性能追踪：这些数据会通过 traces.record(entry) 上报，用于监控和调试。缓存 → 记忆 → 追踪，然后返回 result。
            entry.update(intent=result["intent"], engine=result["engine"],     
                         llm_ms=round((time.perf_counter() - t_llm) * 1000, 1),
                         sources=len(sources),
                         total_ms=round((time.perf_counter() - t_start) * 1000, 1))
            _maybe_cache(query_vec, result, message)  #第二个参数传入的是字典
            memory.add(session_id, message, result["reply"])
            traces.record(entry)
            return result
        except Exception as exc:  # CrewAI 调用失败 -> 降级
            logger.warning("CrewAI 调用失败，降级到内置路由: %s", exc)

    result = await fallback_chat(message, session_id)
    entry.update(intent=result.get("intent"), engine=result.get("engine"),
                 llm_ms=round((time.perf_counter() - t_llm) * 1000, 1),
                 sources=len(result.get("sources", [])),
                 total_ms=round((time.perf_counter() - t_start) * 1000, 1))
    _maybe_cache(query_vec, result, message)
    memory.add(session_id, message, result["reply"])
    traces.record(entry)
    return result
