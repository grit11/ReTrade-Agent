# -*- coding: utf-8 -*-
"""Agent 工具集：RAG 检索问答 / 订单查询(Mock) / 售后规则。
真实项目中：order 工具通过 Feign/HTTP 调用 order-server，售后工具对接多模态质检服务。

命名约定：
- search_knowledge / query_order / after_sale_rule：原始函数，内置路由与 SSE 流式直接调用；
- *_tool 变体：CrewAI Tool 包装（仅安装 crewai 后存在），供多 Agent 编排使用。
"""
import json
import re

from app.rag.retriever import answer_with_rag, build_llm

def search_knowledge(query: str) -> str:
    """检索平台知识库并基于资料回答问题（RAG）。"""
    answer, sources = answer_with_rag(query, build_llm())
    # CrewAI 工具的返回值必须是可传递给下一个 Agent 的内容。
    # 原来只返回 answer，导致 RAG 的 sources 在这里被丢弃，
    # Executive 最终只能生成 sources=[]。将答案和来源一起编码为 JSON，
    # 让 Executive 可以读取并复制到 CrewAnswer.sources。
    # json.dumps = dump to string，把 Python 对象转成 JSON 字符串。反过来是 json.loads（load from string）。
    return json.dumps(  
        {
            "answer": answer,
            "sources": list(sources or []),
        },
        ensure_ascii=False, #为了让中文保持正常显示
    )
    """
    CrewAI 的工具消息中就会变成：
    {
        "role": "tool",
        "name": "search_knowledge",
        "content": "{\"answer\":\"发布商品很简单……\",\"sources\":[\"knowledge_base.md#3\"]}"
    }

    # JSON 字符串的完整流转
        search_knowledge 工具
            │  返回: '{"answer": "七天无理由退货...", "sources": ["售后政策.md"]}'
            ▼
        Executive Agent 看到这段 JSON 文本
            │  从中提取 answer 和 sources
            ▼
        CrewAnswer(reply="七天无理由退货...", sources=["售后政策.md"])
    
    然后 crew.py 中兜底防止agent没有返回sources可以这样解析：
    tool_data = json.loads(content)
    tool_sources = tool_data.get("sources", [])

    """

def query_order(message: str) -> str:
    """查询用户订单状态与物流信息（Mock 数据）。"""
    match = re.search(r"\d{6,}", message)
    order_no = match.group(0) if match else "202608090001"
    return (
        f"订单 {order_no}：状态=已发货，物流=顺丰速运 SF1234567890，"
        "预计 8 月 11 日送达。如需退款或售后，请在订单详情页申请。"
    )


def after_sale_rule(_message: str) -> str:
    """查询平台售后与退货规则。"""
    return ("售后规则：签收 7 天内可申请无理由退货（需不影响二次销售）；"
            "商品破损或与描述不符的，运费由卖家承担；请上传凭证由 AI 质检确认。")


# CrewAI 工具版（供 Agent 编排；未安装 crewai 时为 None，系统自动降级）
search_knowledge_tool = None
query_order_tool = None
after_sale_rule_tool = None
CREW_TOOLS_READY = True
try:
    from crewai.tools import tool
    search_knowledge_tool = tool("search_knowledge")(search_knowledge)
    query_order_tool = tool("query_order")(query_order)
    after_sale_rule_tool = tool("after_sale_rule")(after_sale_rule)
    CREW_TOOLS_READY = True
except Exception:  # pragma: no cover - crewai 未安装
    CREW_TOOLS_READY = False
