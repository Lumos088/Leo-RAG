"""Conversation-memory preparation for history-aware retrieval and generation."""

from __future__ import annotations

from typing import Any


RECENT_MESSAGE_LIMIT = 8
SUMMARY_MAX_CHARS = 3000
REFERENCE_MARKERS = (
    "它", "这个", "这件事", "上述", "上面", "刚才", "之前", "前者", "后者",
    "该方法", "这种", "其中", "继续", "详细说说", "为什么", "那", "其",
    "it", "this", "that", "those", "former", "latter", "above", "continue",
)


def _content(message: dict[str, Any]) -> str:
    return str(message.get("content", "")).strip()


def format_history(messages: list[dict[str, Any]]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels.get(str(item.get('role')), 'Message')}: {_content(item)}"
        for item in messages
        if _content(item)
    )


def needs_context_rewrite(question: str, history: list[dict[str, Any]]) -> bool:
    if not history:
        return False
    normalized = question.strip().lower()
    if any(marker in normalized for marker in REFERENCE_MARKERS):
        return True
    # Very short follow-ups commonly omit the subject, e.g. “优缺点呢？”.
    return len(normalized) <= 14


class ConversationMemory:
    """Produces a compact memory and a standalone query without becoming a fact source."""

    def __init__(self, client=None, model: str = "deepseek-chat"):
        self.client = client
        self.model = model

    def _complete(self, system: str, user: str, max_tokens: int) -> str:
        if self.client is None:
            raise RuntimeError("LLM client unavailable")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _fallback_summary(existing: str, messages: list[dict[str, Any]]) -> str:
        parts = [existing.strip()] if existing.strip() else []
        parts.extend(
            f"{'用户' if item.get('role') == 'user' else '助手'}：{_content(item)[:280]}"
            for item in messages
            if _content(item)
        )
        return "\n".join(parts)[-SUMMARY_MAX_CHARS:]

    def summarize(self, existing: str, messages: list[dict[str, Any]]) -> str:
        transcript = format_history(messages)
        try:
            result = self._complete(
                "压缩对话记忆。只保留用户目标、已确认事实、术语指代和未解决问题；不得添加新事实。",
                f"已有摘要：\n{existing or '（无）'}\n\n新增对话：\n{transcript}\n\n输出更新后的简洁摘要。",
                700,
            )
            return result[:SUMMARY_MAX_CHARS] if result else self._fallback_summary(existing, messages)
        except Exception:
            return self._fallback_summary(existing, messages)

    @staticmethod
    def _last_user_question(history: list[dict[str, Any]]) -> str:
        for item in reversed(history):
            if item.get("role") == "user" and _content(item):
                return _content(item)
        return ""

    def rewrite(self, question: str, summary: str, recent: list[dict[str, Any]]) -> str:
        if not needs_context_rewrite(question, recent):
            return question.strip()
        transcript = format_history(recent)
        try:
            result = self._complete(
                "把用户追问改写成可独立检索的问题。只消解指代，不回答问题，不补充对话中没有的信息，只输出改写结果。",
                f"对话摘要：\n{summary or '（无）'}\n\n最近对话：\n{transcript}\n\n当前追问：{question}",
                180,
            )
            if result and len(result) <= 500:
                return result
        except Exception:
            pass
        anchor = self._last_user_question(recent)
        return f"关于“{anchor}”，{question.strip()}" if anchor else question.strip()

    def prepare(
        self,
        question: str,
        messages: list[dict[str, Any]],
        existing_summary: str = "",
        summarized_through_message_id: int | None = None,
    ) -> dict[str, Any]:
        recent = messages[-RECENT_MESSAGE_LIMIT:]
        older = messages[:-RECENT_MESSAGE_LIMIT]
        unsummarized = [
            item for item in older
            if int(item.get("id") or 0) > int(summarized_through_message_id or 0)
        ]
        summary = existing_summary or ""
        through = summarized_through_message_id
        updated = False
        if unsummarized:
            summary = self.summarize(summary, unsummarized)
            through = int(unsummarized[-1]["id"])
            updated = True

        retrieval_query = self.rewrite(question, summary, recent)
        context_parts = []
        if summary:
            context_parts.append(f"Earlier conversation summary:\n{summary}")
        if recent:
            context_parts.append(f"Recent conversation:\n{format_history(recent)}")
        return {
            "retrieval_query": retrieval_query,
            "summary": summary,
            "summary_updated": updated,
            "summarized_through_message_id": through,
            "recent_messages": recent,
            "conversation_context": "\n\n".join(context_parts),
        }
