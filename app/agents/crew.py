# -*- coding: utf-8 -*-
"""CrewAI 多智能体：意图识别官 -> 客服执行员（多工具编排）。
结构：Agent(角色/目标/背景/工具/LLM) + Task + Crew(sequential)，
与面试叙事一致：把 C2C 客服的"意图路由+工具调用"升级为多 Agent 协作。
"""
import json
import sys
import os
# 关键：把项目根目录加入 sys.path，才能 import app.xxx
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.config import settings
from rich import print as rprint
from app.schemas import CrewAnswer, IntentResult, CrewResult


def run_crew(message: str, history_text: str = "") -> CrewResult:
    from crewai import Agent, Crew, LLM, Process, Task

    from app.agents.tools import (
        after_sale_rule_tool,
        query_order_tool,
        search_knowledge_tool,
    )

    llm = LLM(
        model=f"openai/{settings.llm_model}",
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0.3,
    )
    history_desc = f"\n对话历史：\n{history_text}" if history_text else ""

    router = Agent(
        role="意图识别官",
        goal="准确判断用户消息的意图类型",
        backstory=("你是二手交易平台的意图识别专家，"
                   "只输出 JSON：{{'intent': 'knowledge|order|after_sale_rule|chat', 'reason': '简短理由'}}"),
        llm=llm,
        verbose=False,
    )

    executive = Agent(
        role="客服执行员",
        goal="根据意图调用对应工具，给出真实、友好、简洁的中文答复",
        backstory=("你是二手交易平台客服，擅长用工具查订单、查知识库、讲售后规则。"
                   "必须依据工具返回的真实数据作答，禁止编造。"),
        tools=[search_knowledge_tool, query_order_tool, after_sale_rule_tool],
        llm=llm,
        verbose=False,
    )

    task_router = Task(
        description=f"分析用户消息：{message}（如有对话历史请结合上下文）{history_desc}。只输出意图 JSON。",
        expected_output="JSON：intent / reason",
        output_pydantic=IntentResult, #增加结构化输出，获取intent和reason字段
        agent=router,
    )
    task_exec = Task(
        description=(
            "必须读取上一个 Router 任务输出的 IntentResult，并严格按照 intent 执行："
            "intent=knowledge 时，必须调用 search_knowledge 工具，"
            "禁止直接凭模型常识回答；"
            "intent=order 时，必须调用 query_order 工具；"
            "intent=after_sale_rule 时，必须调用 after_sale_rule 工具；"
            "intent=chat 时才可以直接回复。"
            "最终将工具结果整理成中文答复。"
            f"{history_desc}"
        ),
        expected_output="输出包含 reply 和 sources 字段的结构化结果；sources 必须复制 search_knowledge 返回的来源列表",
        output_pydantic=CrewAnswer, #增加结构化输出，获取reply和sources字段
        context = [task_router],#它显式告诉 CrewAI：当前任务依赖 Router 任务的输出。但需要注意，提示词只能提高工具调用概率，不能百分之百保证 LLM 一定调用工具。
        agent=executive,
    )

    crew = Crew(
        agents=[router, executive],
        tasks=[task_router, task_exec],
        process=Process.sequential,
        verbose=False,
    )
    # result = crew.kickoff()  #返回一个 CrewOutput 对象包含raw(str类型，最终文本的输出)、tasks_output(list/dict类型，每个任务的输出)、token_usage(统计信息)
    # # 此处存在一个bug，只返回raw字段，其他字段未使用，无法获得intent信息
    # #return str(getattr(result, "raw", result)) #启动 Agent 团队执行任务，拿到输出对象；优先取其 raw 原始结果，取不到就用对象本身，最终保证返回字符串。
    # return result

    crew_output = crew.kickoff()
    #rprint(crew_output)

    router_output = crew_output.tasks_output[0]
    exec_output = crew_output.tasks_output[-1]
    """
    虽然对每个任务进行了结构化输出，但是并不会让任务直接返回 IntentResult。CrewAI 仍然会返回一个 TaskOutput 包装对象：
    CrewOutput
    └── tasks_output
        ├── TaskOutput
        │   ├── raw
        │   ├── json_dict
        │   └── pydantic
        └── TaskOutput
    真正的结构化对象在：router_output.pydantic。.pydantic 是从任务结果包装对象中取出已经解析好的 Pydantic 数据，并不是重复结构化。
    """
    router_data = router_output.pydantic
    answer_data = exec_output.pydantic

    # Executive 可能正确生成 reply，但没有把工具返回的 sources 复制到CrewAnswer.sources。此时从 TaskOutput.messages 中读取真实的工具消息，
    # 避免 chat.py 最终只能拿到空来源列表。
    tool_sources = []
    for msg in (getattr(exec_output, "messages", None) or []):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        if msg.get("name") != "search_knowledge":
            continue
        content = msg.get("content", "")
        try:
            tool_data = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(tool_data, dict) and isinstance(tool_data.get("sources"), list):
            tool_sources = [str(item) for item in tool_data["sources"]]
            break

    sources = list(answer_data.sources or tool_sources)

    #CrewAI 不会自动把两个任务合并成一个 CrewResult。必须先分别取出两个任务的数据，再组装：
    return CrewResult(
        reply=answer_data.reply,
        intent=router_data.intent,
        reason=router_data.reason,
        sources=sources,
    )
    #return crew_output
if __name__ == "__main__":
    
    rprint(run_crew("介绍一下你们平台"))
