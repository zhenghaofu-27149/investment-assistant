import os
import json
from typing import Dict, List

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 加载 .env 中的环境变量
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在 .env 文件中配置")

app = FastAPI(title="Investment CRO API", version="0.6.0")

# 允许前端跨域访问（后面你做网页会用到）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本地开发先全开，之后可以收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluateRequest(BaseModel):
    asset_name: str
    research_text: str
    thesis_text: str
    plan_text: str
    emotion_score: int  # 1-10
    capital_impact: str
    position_pct: float  # 仓位百分比


class DimensionScore(BaseModel):
    score: int
    comment: str


class EvaluateResponse(BaseModel):
    overall_score: int
    verdict: str
    dimensions: Dict[str, DimensionScore]
    flags: List[str]
    advice: str
    followup_questions: List[str]


SYSTEM_PROMPT = """
你是用户的首席投资风控官（CRO），冷静、理性、直截了当。

⚠️ 从现在开始，你的输出**必须是合法 JSON**，不能包含任何额外文字、说明、markdown 标题、自然语言前后缀。
如果你输出非 JSON，本系统会直接报错，用户什么也看不到。

你的职责：
- 根据用户提供的研究记录和操作计划，评估这次投资决策是否理性
- 按维度打分
- 标记关键风险
- 提出具体、可执行的改进建议
- 设计 3-8 个尖锐的追问问题，帮助用户在下一轮补充研究

评分维度：
1. research_quality：研究完整度（是否真正理解商业模式、关键指标与竞争格局）
2. logic_rigour：逻辑严谨度（买入/卖出逻辑是否自洽，有没有明显漏洞或自相矛盾）
3. risk_management：风险管理（仓位、分批、止损/卖出计划是否清晰且可执行）
4. emotion_control：情绪控制（是否存在明显的 FOMO、从众或恐慌驱动）
5. capital_safety：资金安全性（这笔钱亏光对生活/重要计划的影响程度是否可接受）

评分规则：
- 80-100：准备充分
- 60-79：勉强可以，但有明显可改进之处
- 40-59：存在较大短板
- 0-39：严重不合理

综合给出 overall_score（0-100）及 verdict：
- "优秀"
- "一般"
- "危险"

flags 中列出 1-5 个最关键的风险点标签，例如：
- "研究不足"
- "买入逻辑不清晰"
- "缺少退出计划"
- "仓位过重"
- "情绪过于激动"
- "资金安全边际不足"

followup_questions：
- 给出 3-8 个问题
- 每个问题都要具体、尖锐，能直接暴露决策中的盲点或假设
- 优先围绕：能力圈、关键假设、最坏情况、仓位合理性、卖出条件

⚠️ 输出格式要求（非常重要）：
你必须严格输出**纯 JSON**，结构如下，不要有任何解释文字、注释、前缀或多余文本：

{
  "overall_score": 0,
  "verdict": "优秀",
  "dimensions": {
    "research_quality": { "score": 0, "comment": "..." },
    "logic_rigour": { "score": 0, "comment": "..." },
    "risk_management": { "score": 0, "comment": "..." },
    "emotion_control": { "score": 0, "comment": "..." },
    "capital_safety": { "score": 0, "comment": "..." }
  },
  "flags": ["..."],
  "advice": "...",
  "followup_questions": ["...", "..."]
}
"""


def build_user_prompt(req: EvaluateRequest) -> str:
    return f"""
标的名称：{req.asset_name}

一、我的研究总结：
{req.research_text}

二、我的买入/卖出逻辑（为什么是现在、核心假设是什么）：
{req.thesis_text}

三、入场 / 分批 / 止损 / 卖出计划：
{req.plan_text}

四、当前情绪自评（1-10，1 非常冷静，10 情绪拉满）：{req.emotion_score}

五、这笔钱亏光的影响：
{req.capital_impact}

六、这次计划仓位（占总资金 %）：{req.position_pct}%
"""


async def call_deepseek(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        # 强制要求 JSON 输出（DeepSeek 兼容 OpenAI 协议）
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"调用 DeepSeek 失败: {resp.status_code} - {resp.text}",
        )

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"解析 DeepSeek 返回失败: {e} - 原始响应: {data}",
        )


def parse_json_safely(raw_text: str) -> dict:
    """
    DeepSeek 偶尔会在 JSON 外面包一层文字或 ```json ```，
    这里尽量从中间把 JSON 抠出来。
    """
    raw_text = raw_text.strip()

    # 1. 先尝试直接解析
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 2. 去掉可能的 ```json 包裹
    if raw_text.startswith("```"):
        stripped = raw_text.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 3. 截取第一个 { 到 最后一个 } 之间
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw_text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    # 打印一份出来方便你在终端看原始返回
    print("==== RAW MODEL OUTPUT BEGIN ====")
    print(raw_text)
    print("==== RAW MODEL OUTPUT END ====")

    # 实在不行就抛错，把前 200 字符返回给你看
    raise HTTPException(
        status_code=500,
        detail=f"模型返回不是合法 JSON：{raw_text[:200]}",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """
    使用 DeepSeek 对本次投资决策进行评估，
    返回：
    - 总分
    - 维度打分与点评
    - 风险标签
    - 建议
    - 模型追问问题
    """
    user_prompt = build_user_prompt(req)
    raw_text = await call_deepseek(user_prompt)
    data = parse_json_safely(raw_text)

    dims = data.get("dimensions") or {}

    def to_dim(key: str) -> DimensionScore:
        item = dims.get(key) or {}
        return DimensionScore(
            score=int(item.get("score", 0)),
            comment=str(item.get("comment", "")),
        )

    flags = data.get("flags") or []
    followups = data.get("followup_questions") or []

    return EvaluateResponse(
        overall_score=int(data.get("overall_score", 0)),
        verdict=str(data.get("verdict", "")),
        dimensions={
            "research_quality": to_dim("research_quality"),
            "logic_rigour": to_dim("logic_rigour"),
            "risk_management": to_dim("risk_management"),
            "emotion_control": to_dim("emotion_control"),
            "capital_safety": to_dim("capital_safety"),
        },
        flags=[str(f) for f in flags],
        advice=str(data.get("advice", "")),
        followup_questions=[str(q).strip() for q in followups],
    )
