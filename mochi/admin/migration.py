"""Chat migration — import chat history from ChatGPT into MochiBot.

Parses exported JSON, preprocesses conversations to reduce noise,
uses an LLM to extract soul/user/core_memory/memory_items,
then writes the user-confirmed results into MochiBot data stores.

NOTE: In-memory session/job storage assumes single Uvicorn worker,
which is the default for the admin portal.
"""

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_SESSION_TTL = 3600   # 1 hour
_JOB_TTL = 7200       # 2 hours

# In-memory stores (single-worker assumption — see module docstring)
_sessions: dict[str, dict] = {}
_jobs: dict[str, dict] = {}

_DATA_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "prompts"

# Rough context window sizes for popular models (used for frontend warnings)
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "claude-3-5-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "deepseek-chat": 65_536,
    "deepseek-v3": 65_536,
    "deepseek-reasoner": 65_536,
}

_EXTRACTION_SYSTEM_PROMPT = """\
你是一个记忆提取专家。用户将提供一段从其他AI助手导出的聊天记录。
请仔细阅读这些对话，从中提取以下四项信息，以 JSON 格式返回。

1. soul — AI助手在对话中展现出的性格特点、说话风格、语气偏好。
   写成一段自然语言描述，可以直接用于定义新AI助手的人格。
   如果对话中看不出明显的性格特征，返回空字符串。

2. user_profile — 关于用户（对话中的人类参与者）的个人信息摘要：
   背景、习惯、职业、兴趣、性格特点等。
   写成一段自然语言描述。如果信息不足，返回空字符串。

3. core_memory — 最重要的核心事实，简洁摘要，100-200字。
   用 bullet point 格式，每条以"- "开头。

4. memory_items — 从对话中提取的所有具体事实、偏好、习惯等。
   每条包含：
   - category: 分类（如"偏好"、"事实"、"习惯"、"目标"、"关系"等）
   - content: 具体内容，一句话概括
   - importance: 重要程度 1（低）/ 2（中）/ 3（高）

请只返回 JSON，不要有任何其他文字或 markdown 标记。格式：
{"soul":"...","user_profile":"...","core_memory":"...","memory_items":[{"category":"偏好","content":"...","importance":2}]}
"""


# ── Data Structures ────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    session_id: str
    conversation_count: int
    raw_message_count: int
    filtered_message_count: int
    estimated_tokens: int


# ── Cleanup ────────────────────────────────────────────────────────────────

def _cleanup_stale(store: dict, ttl: float) -> None:
    now = time.time()
    expired = [k for k, v in store.items() if now - v.get("_ts", 0) > ttl]
    for k in expired:
        del store[k]


# ── ChatGPT Export Parsing ─────────────────────────────────────────────────

def parse_chatgpt_export(raw_bytes: bytes) -> list[dict]:
    """Parse ChatGPT export JSON bytes into a list of conversations."""
    try:
        data = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"无法解析 JSON 文件：{e}") from e

    if not isinstance(data, list):
        raise ValueError("JSON 格式错误：顶层应为数组（对话列表）")

    if not data:
        raise ValueError("JSON 文件中没有对话记录")

    return data


def _traverse_conversation(mapping: dict) -> list[dict]:
    """Reconstruct linear message chain from a ChatGPT conversation mapping.

    The mapping is a DAG (parent/children links). We follow the main branch
    by always picking the last child (matching ChatGPT's display behavior
    when responses are regenerated).
    """
    if not mapping:
        return []

    # Find root node(s) — those with no parent
    roots = [nid for nid, node in mapping.items()
             if node.get("parent") is None or node.get("parent") not in mapping]
    if not roots:
        return []

    messages = []
    current = roots[0]
    visited = set()

    while current and current not in visited:
        visited.add(current)
        node = mapping.get(current, {})
        msg = node.get("message")

        if msg and msg.get("content"):
            parts = msg["content"].get("parts", [])
            # Join only string parts (skip image/code interpreter objects)
            text = "\n".join(p for p in parts if isinstance(p, str)).strip()
            if text:
                role = (msg.get("author") or {}).get("role", "unknown")
                messages.append({
                    "role": role,
                    "content": text,
                    "create_time": msg.get("create_time"),
                })

        # Follow the main branch (last child = latest regeneration)
        children = node.get("children", [])
        current = children[-1] if children else None

    return messages


# ── Preprocessing ──────────────────────────────────────────────────────────

def _code_density(text: str) -> float:
    """Fraction of text that is inside fenced code blocks."""
    blocks = re.findall(r"```[\s\S]*?```", text)
    code_chars = sum(len(b) for b in blocks)
    return code_chars / max(len(text), 1)


def preprocess(conversations: list[dict]) -> PreprocessResult:
    """Apply rule-based filtering and build a transcript for LLM extraction.

    Returns a PreprocessResult with stats and stores the transcript in
    _sessions for later retrieval by session_id.
    """
    _cleanup_stale(_sessions, _SESSION_TTL)

    raw_msg_count = 0
    kept_msgs = []
    transcript_parts = []

    for conv in conversations:
        title = conv.get("title", "无标题")
        mapping = conv.get("mapping", {})
        messages = _traverse_conversation(mapping)
        raw_msg_count += len(messages)

        # Filter messages
        conv_kept = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]

            # Drop system/tool messages
            if role in ("system", "tool"):
                continue
            # Drop very short user messages
            if role == "user" and len(text) < 8:
                continue
            # Drop very long assistant replies (likely verbose explanations)
            if role == "assistant" and len(text) > 500:
                continue

            conv_kept.append(msg)

        if not conv_kept:
            continue

        # Drop code-heavy conversations
        full_text = "\n".join(m["content"] for m in conv_kept)
        if _code_density(full_text) > 0.4:
            continue

        kept_msgs.extend(conv_kept)

        # Build transcript segment
        lines = [f"[对话: {title}]"]
        for msg in conv_kept:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['content']}")
        transcript_parts.append("\n".join(lines))

    transcript = "\n---\n".join(transcript_parts)
    # Conservative token estimate for mixed Chinese/English
    estimated_tokens = len(transcript) // 3

    session_id = uuid.uuid4().hex[:16]
    _sessions[session_id] = {
        "_ts": time.time(),
        "transcript": transcript,
    }

    return PreprocessResult(
        session_id=session_id,
        conversation_count=len(conversations),
        raw_message_count=raw_msg_count,
        filtered_message_count=len(kept_msgs),
        estimated_tokens=estimated_tokens,
    )


# ── Context Window Estimation ──────────────────────────────────────────────

def estimate_context_fit(model_id: str, token_count: int) -> dict:
    """Check if estimated tokens fit within the model's context window.

    Returns {fits: bool, context_window: int|None, pct: float|None}.
    """
    model_lower = model_id.lower()
    for key, ctx in MODEL_CONTEXT_WINDOWS.items():
        if key in model_lower:
            pct = token_count / ctx
            return {"fits": pct < 0.8, "context_window": ctx, "pct": round(pct, 2)}
    return {"fits": True, "context_window": None, "pct": None}


# ── LLM Extraction (background thread) ────────────────────────────────────

def start_extract_job(session_id: str, model_name: str) -> str:
    """Start a background extraction job. Returns job_id for polling."""
    _cleanup_stale(_jobs, _JOB_TTL)

    session = _sessions.get(session_id)
    if not session:
        raise KeyError("Session 不存在或已过期，请重新上传文件")

    job_id = uuid.uuid4().hex[:16]
    _jobs[job_id] = {
        "_ts": time.time(),
        "status": "running",
        "result": None,
        "error": None,
    }

    t = threading.Thread(
        target=_run_extract,
        args=(job_id, session_id, model_name),
        daemon=True,
    )
    t.start()
    return job_id


def _run_extract(job_id: str, session_id: str, model_name: str) -> None:
    """Run LLM extraction in a background thread."""
    try:
        transcript = _sessions[session_id]["transcript"]

        # Get model credentials from DB (unmasked)
        from mochi.admin.admin_db import get_model
        entry = get_model(model_name, mask_key=False)
        if not entry:
            raise ValueError(f"模型 '{model_name}' 未找到")

        # Build a one-off LLM client (not through model pool —
        # migration is a one-time operation where the user picks a
        # specific model, not a tier)
        from mochi.llm import _make_client
        client = _make_client(
            provider=entry["provider"],
            api_key=entry["api_key"],
            model=entry["model"],
            base_url=entry.get("base_url", ""),
        )

        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        response = client.chat(messages, temperature=0.2, max_tokens=4096)
        content = response.content.strip()

        # Parse JSON — handle possible markdown fences
        parsed = _parse_llm_json(content)

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = parsed
        log.info("Migration extraction job %s completed", job_id)

    except Exception as e:
        log.exception("Migration extraction job %s failed", job_id)
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)[:1000]


def _parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code fences."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fence
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the outermost {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("LLM 返回的内容无法解析为 JSON，请重试或换用其他模型")


def get_job_status(job_id: str) -> dict:
    """Get the status of an extraction job."""
    job = _jobs.get(job_id)
    if not job:
        raise KeyError("任务不存在或已过期")
    return {
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
    }


# ── Apply Migration Results ────────────────────────────────────────────────

def apply_migration(payload: dict) -> dict:
    """Write user-confirmed extraction results into MochiBot data stores.

    This function is synchronous (calls embed() which is blocking HTTP).
    Route handlers should call it via asyncio.to_thread().
    """
    from mochi.config import OWNER_USER_ID
    from mochi import db
    from mochi.prompt_loader import reload_all
    from mochi.model_pool import get_pool

    uid = OWNER_USER_ID or 0
    stats = {
        "soul_written": False,
        "user_written": False,
        "core_memory_written": False,
        "memory_items_imported": 0,
    }

    # ── Soul ──
    soul = (payload.get("soul") or "").strip()
    if soul:
        path = _DATA_PROMPTS_DIR / "system_chat" / "soul.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(soul, encoding="utf-8")
        stats["soul_written"] = True

    # ── User profile ──
    user_profile = (payload.get("user_profile") or "").strip()
    if user_profile:
        path = _DATA_PROMPTS_DIR / "system_chat" / "user.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(user_profile, encoding="utf-8")
        stats["user_written"] = True

    # Reload prompts once if either was written
    if stats["soul_written"] or stats["user_written"]:
        reload_all()

    # ── Core memory ──
    core_memory = (payload.get("core_memory") or "").strip()
    if core_memory:
        db.update_core_memory(uid, core_memory)
        stats["core_memory_written"] = True

    # ── Memory items ──
    items = payload.get("memory_items") or []
    pool = get_pool()
    for item in items:
        if not item.get("selected", True):
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue
        category = item.get("category", "其他")
        importance = max(1, min(3, int(item.get("importance", 1))))
        embedding = pool.embed(content)
        db.save_memory_item(
            uid,
            category=category,
            content=content,
            importance=importance,
            source="migration",
            embedding=embedding,
        )
        stats["memory_items_imported"] += 1

    return {"ok": True, **stats}
