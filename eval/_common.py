"""端到端 / agentic RAG 评估的公共件:env 加载、demo 库拷贝(避开 live MCP 单 client 锁)、
DeepSeek 文本/JSON 客户端。被 gen_gold / run_eval / acl_regression 复用。

**为何拷贝 demo 库**:嵌入式 Qdrant 单 client 独占锁。你在 Claude Code 里连着的 MCP server 正持有 ~/rag_demo/qdrant 锁,
评估脚本若直开同目录会 "already accessed by another instance"。统一先 copytree 到临时工作目录再开,既不抢锁也不污染原库。
"""
from __future__ import annotations

import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)   # = pharos 仓根;load_env 读 REPO/.env(即 pharos/.env)。引擎包已 pip install -e . 安装

DEMO_SRC = os.path.expanduser("~/rag_demo")
DEMO_COLLECTION = "demo"
DENSE_DIM = 1024

# 评估库源(可 env 覆盖指向更大的库,如 index_eval_corpus.py 建的 ~/rag_eval_big / collection=evalbig)
EVAL_SRC = os.environ.get("PHAROS_EVAL_SRC") or os.environ.get("RAG_EVAL_SRC", DEMO_SRC)
EVAL_COLLECTION = os.environ.get("PHAROS_EVAL_COLLECTION") or os.environ.get("RAG_EVAL_COLLECTION", DEMO_COLLECTION)

# 评估三档模型(全走 DeepSeek;同厂自评的循环偏差见 README "诚实警告")。RAG_EVAL_* 留一版弃用别名。
# ⚠ PHAROS_EVAL_GEN_MODEL 须与生产 PHAROS_LLM_MODEL 一致 —— 否则测的不是实际部署配置。
GEN_MODEL = os.environ.get("PHAROS_EVAL_GEN_MODEL") or os.environ.get("RAG_EVAL_GEN_MODEL", "deepseek-v4-flash")
JUDGE_MODEL = os.environ.get("PHAROS_EVAL_JUDGE_MODEL") or os.environ.get("RAG_EVAL_JUDGE_MODEL", "deepseek-v4-flash")


def load_env(path: str | None = None) -> None:
    """极简 .env 加载(免 python-dotenv);已存在的环境变量优先(setdefault)。"""
    path = path or os.path.join(REPO, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def copy_demo(dest: str, src: str | None = None) -> tuple[str, str]:
    """把评估库源(默认 EVAL_SRC,可 env PHAROS_EVAL_SRC 覆盖)拷到 dest(qdrant + sidecar),返回 (qdrant_path, sidecar_dir)。
    不抢 live 锁、不污染原库。"""
    src = src or EVAL_SRC
    if not os.path.isdir(src):
        raise FileNotFoundError(f"评估库不存在:{src};先跑 index_demo.py 或 index_eval_corpus.py")
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest)
    return os.path.join(dest, "qdrant"), os.path.join(dest, "sidecar")


def make_retriever(work: str):
    """在评估库副本上建 Retriever(GPU 首查 lazy load Qwen3-VL 8B)。库与 collection 由 EVAL_SRC/EVAL_COLLECTION 决定。"""
    from embedder import EmbedConfig, Retriever
    qpath, sdir = copy_demo(work)
    cfg = EmbedConfig(qdrant_path=qpath, sidecar_dir=sdir, dense_dim=DENSE_DIM, collection=EVAL_COLLECTION)
    return Retriever(cfg), cfg


def demo_user():
    """demo 库全 tenant=demo/public —— 用此身份即可见全部 4 篇。"""
    from embedder import User
    return User(tenant="demo", principals=[])


def scroll_chunks(qdrant_path: str, collection: str = EVAL_COLLECTION) -> list[dict]:
    """扫一个 Qdrant 库的所有 point payload(CPU,无需 GPU)。给 gen_gold 采样用。"""
    from qdrant_client import QdrantClient
    client = QdrantClient(path=qdrant_path)
    out, offset = [], None
    while True:
        pts, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
        out.extend(p.payload for p in pts)
        if offset is None:
            break
    return out


# ---------------- DeepSeek 客户端 ----------------

class TextLLM:
    """被测系统用的生成 LLM(真实服务配置:thinking off / temp 0)。直接复用 generator.OpenAICompatibleLLM。"""

    def __init__(self, model: str = GEN_MODEL):
        from generator import OpenAICompatibleLLM
        self.impl = OpenAICompatibleLLM(model=model, thinking=False)

    def complete(self, messages) -> str:
        return self.impl.complete(messages)


class JsonLLM:
    """gold 生成 + 裁判用:DeepSeek JSON 模式(response_format json_object)+ 一次重试。返回 dict。

    thinking 显式关(V4 Flash 思考默认可能开,不关会要求回传 reasoning_content 而 400);json_object 强约束输出可解析。
    """

    def __init__(self, model: str = JUDGE_MODEL, temperature: float = 0.0, max_tokens: int = 1200):
        from openai import OpenAI
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("缺 DEEPSEEK_API_KEY(放 .env,勿提交)")
        self.client = OpenAI(base_url="https://api.deepseek.com", api_key=key, timeout=120)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def ask(self, system: str, user: str) -> dict:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=self.temperature,
                max_tokens=self.max_tokens, response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}})
            content = resp.choices[0].message.content or ""
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                if attempt == 0:
                    msgs.append({"role": "assistant", "content": content})
                    msgs.append({"role": "user", "content": "上一条不是合法 JSON。只输出一个合法 JSON 对象,不要任何额外文字。"})
                    continue
                raise ValueError(f"裁判/生成返回非法 JSON(两次):{content[:200]}")
        raise AssertionError("unreachable")
