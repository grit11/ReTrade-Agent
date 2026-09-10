# -*- coding: utf-8 -*-
"""FastAPI 应用入口。

本模块负责把底层能力组装成 HTTP 服务，主要包含：

* 应用生命周期：启动时自动导入示例知识库；
* HTTP 中间件：请求耗时日志、响应耗时响应头、按 IP 限流；
* 基础运维接口：``/health``、``/api/v1/stats``、``/api/v1/traces``；
* 知识库接口：``/api/v1/ingest``；
* 对话接口：普通 JSON 对话 ``/api/v1/chat`` 和 SSE 流式对话
  ``/api/v1/chat/stream``；
* 统一异常兜底：记录服务端完整异常，但向客户端返回稳定的错误结构。

需要特别区分两条对话路径：

``/api/v1/chat``
    调用 ``app.services.chat.chat``，由该服务决定先走 CrewAI，还是
    降级到 LangChain 内置路由。

``/api/v1/chat/stream``
    在本文件中直接编排缓存、意图识别、RAG/订单工具和 SSE 事件，当前
    不调用 CrewAI，适合把中间阶段实时展示给前端控制台。
"""
import asyncio
import json
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app.agents.tools import CREW_TOOLS_READY
from app.config import settings
from app.rag.retriever import kb
from app.schemas import ChatRequest, ChatResponse, IngestResponse, StatsResponse
from app.services.chat import CHAT_PROMPT, chat, classify_intent
from app.services.ratelimit import limiter
from app.services.semantic_cache import semantic_cache
from app.services.tracing import traces

BASE_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger("airobot.main")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """管理 FastAPI 应用的启动与关闭生命周期。

    FastAPI 会在应用启动前执行 ``yield`` 之前的代码，在应用关闭时执行
    ``yield`` 之后的代码。本项目目前没有关闭清理逻辑，因此这里主要做
    启动初始化：如果全局知识库为空，就导入 ``data/knowledge_base.md``。

    使用全局 ``kb`` 的原因是检索器、Embedding、BM25 索引和可选重排器都
    需要在进程内复用；如果每个请求重新创建，会导致重复初始化甚至重复
    向量化。导入失败只记录 warning，不阻止健康检查和 API 服务启动，便于
    用户先启动服务、再检查 Embedding 配置。
    """
    sample = BASE_DIR / "data" / "knowledge_base.md"
    if sample.exists() and kb.chunk_count == 0:
        try:
            kb.ingest_file(sample)
        except Exception as exc:
            logger.warning("启动时导入示例知识库失败（请检查 AIROBOT_EMBEDDING_API_KEY）: %s", exc)
    yield


app = FastAPI(
    title="AI Robot 智能客服服务（FastAPI + LangChain RAG + CrewAI 多 Agent）",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个 HTTP 请求的状态和耗时。

    ``call_next`` 会继续调用后续中间件和路由处理器。本函数使用单调高精度
    计时器计算耗时，并将结果写入日志和 ``X-Process-Time-Ms`` 响应头，方便
    网关、浏览器和控制台观察接口延迟。
    """
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path,
                response.status_code, elapsed_ms)
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """在业务接口入口执行按 IP 的滑动窗口限流。

    只有 ``/api/v1`` 下的业务接口参与限流；``stats`` 和 ``traces`` 被
    明确排除，使监控控制台在业务请求过多时仍然可以读取状态。被拦截的
    请求会写入 ``traces`` 并返回 HTTP 429；放行时必须调用 ``call_next``，
    请求才会继续到达目标路由。
    """
    if (settings.ratelimit_enabled and request.url.path.startswith("/api/v1")
            and request.url.path not in ("/api/v1/stats", "/api/v1/traces")):
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.allow(client_ip):
            logger.warning("限流拦截: %s %s from %s", request.method, request.url.path, client_ip)
            traces.record({"message": f"{request.method} {request.url.path}",
                           "session_id": client_ip, "intent": "ratelimited",
                           "status": 429, "cache_checked": False, "total_ms": 0.0})
            return JSONResponse(status_code=429, content={"detail": "请求过于频繁，请稍后再试。"})
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """处理没有被业务代码捕获的异常。

    服务端日志保存完整堆栈，客户端只收到固定的 500 JSON，避免暴露内部
    路径、第三方 SDK 细节或其他敏感信息。流式接口在自己的生成器中捕获
    异常，因此通常会发送 ``stage=error`` 事件，而不是走这里的普通 JSON。
    """
    logger.exception("未处理异常: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "服务内部错误，请稍后重试。"})


@app.get("/health")
def health():
    """返回最小健康检查结果。

    该接口只读取配置，不调用 LLM、Embedding 或知识库，适合容器
    healthcheck、负载均衡探活和启动脚本轮询。``status=ok`` 表示 Web 进程
    可响应，并不保证外部模型服务一定可用。
    """
    return {"status": "ok", "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model}


@app.get("/api/v1/stats", response_model=StatsResponse)
def stats():
    """汇总当前进程的运行状态和关键指标。

    数据来自知识库 ``kb``、语义缓存 ``semantic_cache`` 和限流器 ``limiter``。
    ``crew_available`` 表示 CrewAI 工具是否成功导入，``use_crew`` 表示配置
    开关；普通 ``/api/v1/chat`` 只有在二者同时为真时才会尝试 CrewAI。
    """
    cache = semantic_cache.stats()
    rl = limiter.stats()
    return StatsResponse(
        total_chunks=kb.chunk_count,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        use_crew=settings.use_crew,
        crew_available=CREW_TOOLS_READY,
        hybrid_enabled=settings.hybrid_enabled,
        bm25_ready=kb.bm25_ready,
        rerank_enabled=settings.rerank_enabled,
        cache_enabled=settings.cache_enabled,
        cache_size=cache["size"],
        cache_hits=cache["hits"],
        cache_misses=cache["misses"],
        cache_threshold=cache["threshold"],
        ratelimit_enabled=settings.ratelimit_enabled,
        ratelimit_per_minute=rl["limit_per_minute"],
        ratelimit_blocked=rl["blocked"],
    )


@app.get("/dashboard")
def dashboard():
    """返回内置单页监控控制台。

    页面是仓库中的静态 HTML，无需额外前端构建；浏览器打开后会轮询
    ``/api/v1/stats``、``/api/v1/traces``，并调用流式对话接口展示缓存、
    意图、检索和生成阶段。
    """
    html = (BASE_DIR / "app" / "static" / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/v1/traces")
def traces_api(limit: int = 50):
    """返回最近请求明细和聚合追踪统计。

    ``limit`` 控制返回的最近请求条数；聚合摘要仍基于环形缓冲区中的全部
    记录，包含平均耗时、P95、缓存命中率、限流次数和意图分布。
    """
    return {"entries": traces.recent(limit), "summary": traces.summary()}


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """上传并导入一个知识库文件。

    处理步骤是：检查 Embedding Key；校验扩展名；把上传内容写入临时文件；
    调用 ``kb.ingest_file`` 完成解析、分块、向量入库和 BM25 更新；最后在
    ``finally`` 中删除临时文件并返回本次新增/当前总分块数。默认内存向量库
    的导入结果只对当前服务进程有效。
    """
    if not settings.embedding_api_key:
        raise HTTPException(status_code=400, detail="未配置 AIROBOT_EMBEDDING_API_KEY，无法向量化入库")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".docx", ".md", ".txt", ".markdown"):
        raise HTTPException(status_code=400, detail="仅支持 pdf / docx / md / txt")
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        chunks = kb.ingest_file(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"解析失败: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return IngestResponse(file_name=file.filename or "unknown",
                          chunks=chunks, total_chunks=kb.chunk_count)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_ep(req: ChatRequest):
    """普通 JSON 对话接口的薄适配层。

    真正的编排位于 ``app.services.chat.chat``：包括首轮语义缓存、CrewAI
    优先路径、LangChain fallback、RAG/订单/闲聊、会话记忆和链路追踪。本
    函数只负责把 HTTP 请求模型传入服务，并转换成稳定的响应模型。
    """
    result = await chat(req.message, req.session_id)
    return ChatResponse(
        reply=result["reply"],
        intent=result.get("intent"),
        sources=result.get("sources", []),
        engine=result.get("engine", "langchain"),
        used_crew=result.get("used_crew", False),
        cache_hit=result.get("cache_hit", False),
    )


def _sse(payload: dict) -> str:
    """把字典编码成标准 SSE 数据帧。

    SSE 要求事件以 ``data:`` 开头，并用空行结束；``ensure_ascii=False``
    保证中文不会被编码成 ASCII 转义。所有 stage、intent、token、done 事件
    都通过这里统一格式化。
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式对话接口。

    本接口在当前文件中直接编排 ``限流事件 -> 语义缓存 -> 意图识别 ->
    RAG/订单/闲聊 -> done``。RAG 和闲聊通过 ``chain.astream`` 逐片段发送
    token；订单工具一次性发送一个 token；多轮会话跳过语义缓存。注意：
    这条 SSE 路径当前不调用 CrewAI，而是直接使用 ``classify_intent`` 和
    工具/RAG。
    """

    async def _error(message: str):
        """在无法正常开始流式处理时，发送最小错误事件序列。"""
        yield _sse({"type": "token", "content": message})
        yield _sse({"type": "done"})

    if not settings.llm_api_key:
        return StreamingResponse(_error("未配置 AIROBOT_LLM_API_KEY，请复制 .env.example 为 .env。"),
                                 media_type="text/event-stream")

    from langchain_core.output_parsers import StrOutputParser

    from app.agents.tools import query_order
    from app.rag.retriever import RAG_PROMPT, build_llm
    from app.services.memory import memory

    async def gen():
        """生成一次流式对话的全部 SSE 事件。

        生成器既负责向前端报告中间阶段，也负责请求结束后的会话记忆、语义
        缓存和 TraceRecorder 写入；前端用 ``done`` 事件收束本次请求。
        """
        t_start = time.perf_counter()
        entry = {"message": req.message[:80], "session_id": req.session_id,
                 "status": 200, "engine": "sse"}
        try:
            # 限流中间件已经在进入路由前完成真正检查；这里发送成功事件，
            # 让前端控制台的时间线保持完整。
            yield _sse({"type": "stage", "stage": "rate_limit",
                        "msg": "限流检查通过，请求进入服务", "ms": 0.0, "ok": True})
            history = memory.get_messages(req.session_id)
            query_vec = None
            if settings.cache_enabled and settings.embedding_api_key and not history:
                # Embedding 接口是同步调用，放入线程池，避免阻塞事件循环。
                t_cache = time.perf_counter()
                yield _sse({"type": "stage", "stage": "cache", "msg": "语义缓存查询中…", "ms": 0.0})
                query_vec = await asyncio.to_thread(kb.embed_query, req.message)
                cached = semantic_cache.get(query_vec, req.message)
                cache_ms = round((time.perf_counter() - t_cache) * 1000, 1)
                entry["cache_lookup_ms"] = cache_ms
                entry["cache_checked"] = True
                if cached is not None:
                    entry.update(cache_hit=True, intent=cached.get("intent"), tokens=1,
                                 total_ms=round((time.perf_counter() - t_start) * 1000, 1))
                    traces.record(entry)
                    yield _sse({"type": "stage", "stage": "cache", "msg": "语义缓存命中，直接返回",
                                "ms": cache_ms, "hit": True, "ok": True})
                    yield _sse({"type": "intent", "intent": cached.get("intent")})
                    yield _sse({"type": "token", "content": cached["reply"]})
                    yield _sse({"type": "stage", "stage": "write", "msg": "命中缓存，无需写入",
                                "ms": 0.0, "ok": True})
                    yield _sse({"type": "done", "cache_hit": True, "intent": cached.get("intent"),
                                "sources": cached.get("sources", []),
                                "total_ms": entry["total_ms"]})
                    return
                yield _sse({"type": "stage", "stage": "cache", "msg": "语义缓存未命中", "ms": cache_ms,
                            "hit": False, "ok": True})
            else:
                # 有历史时跳过缓存，避免同一个问题因上下文不同而复用错误答案。
                reason = "未开启" if not (settings.cache_enabled and settings.embedding_api_key) else "多轮会话，跳过缓存"
                yield _sse({"type": "stage", "stage": "cache", "msg": f"跳过语义缓存（{reason}）",
                            "ms": 0.0, "ok": True, "skipped": True})
            # 流式链路直接做意图分类；解析失败时 classify_intent 会兜底为 chat。
            t_intent = time.perf_counter()
            intent = await classify_intent(req.message)
            intent_ms = round((time.perf_counter() - t_intent) * 1000, 1)
            entry["intent_ms"] = intent_ms
            entry["intent"] = intent
            yield _sse({"type": "intent", "intent": intent})
            yield _sse({"type": "stage", "stage": "intent", "msg": f"意图识别为 {intent}",
                        "intent": intent, "ms": intent_ms, "ok": True})

            if intent == "order":
                # 订单是动态数据，不走 RAG，也不写入语义缓存。当前工具是
                # Mock，生产环境可替换为订单服务 HTTP 调用。
                t_tool = time.perf_counter()
                reply = query_order(req.message)
                tool_ms = round((time.perf_counter() - t_tool) * 1000, 1)
                yield _sse({"type": "stage", "stage": "tool", "tool": "query_order",
                            "msg": "调用订单查询工具 query_order", "ms": tool_ms, "ok": True})
                yield _sse({"type": "token", "content": reply})
                yield _sse({"type": "done", "intent": intent,
                            "total_ms": round((time.perf_counter() - t_start) * 1000, 1)})
                memory.add(req.session_id, req.message, reply)
                entry.update(tokens=1, llm_ms=tool_ms,
                             total_ms=round((time.perf_counter() - t_start) * 1000, 1))
                traces.record(entry)
                return

            if intent == "knowledge" and kb.chunk_count > 0:
                # 检索通常包含同步向量/BM25/重排操作，放入线程池避免阻塞事件循环。
                t_retr = time.perf_counter()
                docs, detail = await asyncio.to_thread(kb.search_detailed, req.message)
                retr_ms = round((time.perf_counter() - t_retr) * 1000, 1)
                entry["retrieval_ms"] = retr_ms
                entry["retrieval_detail"] = detail
                sources = [f"{d.metadata.get('title', '')}#{d.metadata.get('chunk', 0)}" for d in docs]
                yield _sse({"type": "stage", "stage": "retrieval", "msg": "混合检索完成",
                            "ms": retr_ms, "detail": detail, "sources": sources, "ok": True})
                context = "\n\n".join(d.page_content for d in docs)
                # RAG Prompt 接收检索上下文、问题和历史；astream 每产出一段
                # 文本，就立即转换成 token 事件发给前端。
                chain = RAG_PROMPT | build_llm() | StrOutputParser()
                parts: list[str] = []
                t_gen = time.perf_counter()
                async for chunk in chain.astream({"context": context, "question": req.message, "history": history}):
                    if chunk:
                        if not parts:
                            entry["first_token_ms"] = round(
                                (time.perf_counter() - t_start) * 1000, 1)
                        parts.append(chunk)
                        yield _sse({"type": "token", "content": chunk})
                llm_ms = round((time.perf_counter() - t_gen) * 1000, 1)
                entry["llm_ms"] = llm_ms
                entry["tokens"] = len(parts)
                yield _sse({"type": "stage", "stage": "generate",
                            "msg": f"RAG 生成完成（{len(parts)} tokens）", "ms": llm_ms,
                            "tokens": len(parts), "ok": True})
                yield _sse({"type": "stage", "stage": "write",
                            "msg": "写入会话记忆与语义缓存", "ms": 0.0, "ok": True})
                yield _sse({"type": "done", "intent": intent, "sources": sources,
                            "total_ms": round((time.perf_counter() - t_start) * 1000, 1)})
                memory.add(req.session_id, req.message, "".join(parts))
                if query_vec is not None:
                    semantic_cache.put(query_vec, {
                        "reply": "".join(parts), "intent": "knowledge",
                        "sources": sources, "engine": "langchain"}, req.message)
                entry.update(sources=len(sources),
                             total_ms=round((time.perf_counter() - t_start) * 1000, 1))
                traces.record(entry)
                return

            # 意图不是 knowledge，或 knowledge 但知识库为空时，走闲聊 Prompt。
            # 这是 SSE 实现的行为；非流式 fallback_chat 对空知识库有专门提示。
            chain = CHAT_PROMPT | build_llm() | StrOutputParser()
            parts = []
            t_gen = time.perf_counter()
            async for chunk in chain.astream({"message": req.message, "history": history}):
                if chunk:
                    if not parts:
                        entry["first_token_ms"] = round(
                            (time.perf_counter() - t_start) * 1000, 1)
                    parts.append(chunk)
                    yield _sse({"type": "token", "content": chunk})
            llm_ms = round((time.perf_counter() - t_gen) * 1000, 1)
            entry["llm_ms"] = llm_ms
            entry["tokens"] = len(parts)
            yield _sse({"type": "stage", "stage": "generate",
                        "msg": f"闲聊生成完成（{len(parts)} tokens）", "ms": llm_ms,
                        "tokens": len(parts), "ok": True})
            yield _sse({"type": "stage", "stage": "write",
                        "msg": "写入会话记忆与语义缓存", "ms": 0.0, "ok": True})
            yield _sse({"type": "done", "intent": intent,
                        "total_ms": round((time.perf_counter() - t_start) * 1000, 1)})
            memory.add(req.session_id, req.message, "".join(parts))
            if query_vec is not None:
                semantic_cache.put(query_vec, {
                    "reply": "".join(parts), "intent": "chat",
                    "sources": [], "engine": "langchain"}, req.message)
            entry.update(total_ms=round((time.perf_counter() - t_start) * 1000, 1))
            traces.record(entry)
        except Exception as exc:
            # 生成器内部捕获异常，保证客户端仍能收到结构化结束事件；服务端
            # 日志保留完整堆栈，方便排查。
            logger.exception("流式对话异常: %s", exc)
            entry.update(status=500,
                         total_ms=round((time.perf_counter() - t_start) * 1000, 1))
            traces.record(entry)
            yield _sse({"type": "stage", "stage": "error", "msg": f"处理失败：{exc}", "ok": False})
            yield _sse({"type": "token", "content": f"服务开小差了：{exc}"})
            yield _sse({"type": "done"})
    return StreamingResponse(gen(), media_type="text/event-stream")
