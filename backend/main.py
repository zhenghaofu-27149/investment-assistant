import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ========== 环境变量 & 基础配置 ==========

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在 .env 文件中配置")

# JWT 相关配置
SECRET_KEY = os.getenv("SECRET_KEY", "change_me_in_.env")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 天

# 数据库配置（本地 SQLite）
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./users.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ========== SQLAlchemy 模型 ==========

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DecisionHistory(Base):
    __tablename__ = "decision_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False, default="default")
    asset_name = Column(String, index=True, nullable=False, default="")
    request_json = Column(Text, nullable=False, default="{}")
    response_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# 第一次启动自动建表
Base.metadata.create_all(bind=engine)

# ========== FastAPI 应用 ==========

app = FastAPI(title="Investment CRO API", version="0.7.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 先全开，后面再收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 安全 & 工具函数 ==========

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证登录状态，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    return user


# ========== Pydantic 模型 ==========

class EvaluateRequest(BaseModel):
    # 关键：前端会发 user_id，用于隔离历史记录
    user_id: str = Field(default="default", max_length=128)

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


class UserCreate(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    created_at: datetime

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class HistoryListRequest(BaseModel):
    user_id: str = "default"
    limit: int = 50


class HistoryClearRequest(BaseModel):
    user_id: str = "default"


class HistoryItemOut(BaseModel):
    id: int
    user_id: str
    asset_name: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    created_at: datetime


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
1. research_quality：研究完整度
2. logic_rigour：逻辑严谨度
3. risk_management：风险管理
4. emotion_control：情绪控制
5. capital_safety：资金安全性

评分规则：
- 80-100：准备充分
- 60-79：勉强可以
- 40-59：短板明显
- 0-39：严重不合理

综合给出 overall_score（0-100）及 verdict：
- "优秀"
- "一般"
- "危险"

⚠️ 输出必须是纯 JSON，结构如下：

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
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    if raw_text.startswith("```"):
        stripped = raw_text.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw_text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    print("==== RAW MODEL OUTPUT BEGIN ====")
    print(raw_text)
    print("==== RAW MODEL OUTPUT END ====")

    raise HTTPException(status_code=500, detail=f"模型返回不是合法 JSON：{raw_text[:200]}")


# ========== 健康检查 ==========

@app.get("/health")
async def health():
    return {"status": "ok"}


# ========== 历史记录接口 ==========

@app.post("/history/list", response_model=List[HistoryItemOut])
def history_list(req: HistoryListRequest, db: Session = Depends(get_db)):
    limit = max(1, min(req.limit, 200))
    rows = (
        db.query(DecisionHistory)
        .filter(DecisionHistory.user_id == req.user_id)
        .order_by(DecisionHistory.created_at.desc())
        .limit(limit)
        .all()
    )

    out: List[HistoryItemOut] = []
    for r in rows:
        try:
            request_obj = json.loads(r.request_json or "{}")
        except Exception:
            request_obj = {}
        try:
            response_obj = json.loads(r.response_json or "{}")
        except Exception:
            response_obj = {}

        out.append(
            HistoryItemOut(
                id=r.id,
                user_id=r.user_id,
                asset_name=r.asset_name,
                request=request_obj,
                response=response_obj,
                created_at=r.created_at,
            )
        )
    return out


@app.post("/history/clear")
def history_clear(req: HistoryClearRequest, db: Session = Depends(get_db)):
    db.query(DecisionHistory).filter(DecisionHistory.user_id == req.user_id).delete()
    db.commit()
    return {"ok": True}


# ========== 核心评估接口 ==========

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest, db: Session = Depends(get_db)) -> EvaluateResponse:
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

    resp_obj = EvaluateResponse(
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

    # 评估结果落库（用于历史记录）
    try:
        db.add(
            DecisionHistory(
                user_id=req.user_id or "default",
                asset_name=req.asset_name or "",
                request_json=json.dumps(req.model_dump(), ensure_ascii=False),
                response_json=json.dumps(resp_obj.model_dump(), ensure_ascii=False),
                created_at=datetime.utcnow(),
            )
        )
        db.commit()
    except Exception as e:
        # 不因为存库失败就让评估失败（否则你会更想砸电脑）
        print("保存历史记录失败：", e)

    return resp_obj


# ========== 用户注册 / 登录接口（保留你的原逻辑） ==========

@app.post("/auth/register", response_model=UserOut)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = get_user_by_email(db, user_in.email)
    if existing:
        raise HTTPException(status_code=400, detail="该邮箱已注册")

    user = User(email=user_in.email, hashed_password=get_password_hash(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="邮箱或密码错误")

    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me", response_model=UserOut)
def read_me(current_user: User = Depends(get_current_user)):
    return current_user
