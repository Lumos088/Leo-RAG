import asyncio
import json

from web_ui.api.conversation_store import ConversationStore
from web_ui.api.memory import ConversationMemory, needs_context_rewrite


def test_conversation_crud_isolation_restart_and_idempotency(tmp_path):
    path = tmp_path / "conversations.db"
    store = ConversationStore(path)
    first = store.create_conversation()
    second = store.create_conversation("Second")
    one = store.add_message(first["id"], "user", "什么是死锁？", request_id="r1")
    duplicate = store.add_message(first["id"], "user", "什么是死锁？", request_id="r1")
    store.add_message(
        first["id"], "assistant", "答案[1]", request_id="r1",
        retrieval=[{"id": "p1", "text": "evidence"}], citations=[{"rank": 1}],
        metadata={"retrieval_query": "什么是死锁？", "use_bm25": True},
    )
    assert one["id"] == duplicate["id"]
    assert store.list_messages(second["id"]) == []
    assert store.get_conversation(first["id"])["title"] == "什么是死锁？"

    reopened = ConversationStore(path)
    messages = reopened.list_messages(first["id"])
    assert len(messages) == 2
    assert messages[1]["retrieval"][0]["id"] == "p1"
    assert messages[1]["citations"] == [{"rank": 1}]
    assert messages[1]["metadata"]["use_bm25"] is True

    reopened.rename_conversation(first["id"], "OS review")
    assert reopened.get_conversation(first["id"])["title"] == "OS review"
    reopened.clear_messages(first["id"])
    assert reopened.list_messages(first["id"]) == []
    reopened.delete_conversation(second["id"])
    assert [item["id"] for item in reopened.list_conversations()] == [first["id"]]


class FakeCompletions:
    def create(self, **kwargs):
        text = "死锁的四个必要条件分别是什么？"
        return type("Response", (), {
            "choices": [type("Choice", (), {"message": type("Message", (), {"content": text})()})]
        })()


class FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_followup_rewrite_and_summary_fallback():
    history = [
        {"id": index + 1, "role": "user" if index % 2 == 0 else "assistant", "content": f"消息 {index + 1}"}
        for index in range(10)
    ]
    assert needs_context_rewrite("它有哪些条件？", history)
    memory = ConversationMemory(FakeClient())
    prepared = memory.prepare("它有哪些条件？", history)
    assert prepared["retrieval_query"] == "死锁的四个必要条件分别是什么？"
    assert prepared["summary_updated"] is True
    assert prepared["summarized_through_message_id"] == 2
    assert len(prepared["recent_messages"]) == 8

    fallback = ConversationMemory(None).prepare("优缺点呢？", history)
    assert "关于“消息 9”" in fallback["retrieval_query"]
    assert fallback["summary"]


class FakeMemory:
    def prepare(self, question, messages, summary, through):
        return {
            "retrieval_query": f"standalone: {question}",
            "conversation_context": "memory",
            "summary": summary,
            "summary_updated": False,
            "summarized_through_message_id": through,
        }


class FakeService:
    def __init__(self, fail=False):
        self.memory = FakeMemory()
        self.fail = fail

    async def query_stream(self, question, mode, **kwargs):
        yield {"type": "retrieval", "retrieval": [{"id": "p1"}], "citations": [{"rank": 1}], "mode": mode, "lang": "zh", "retrieval_query": kwargs["retrieval_question"]}
        yield {"type": "chunk", "content": "partial"}
        if self.fail:
            yield {"type": "error", "message": "failed"}
            return
        yield {"type": "done", "answer": "partial"}


def _decode_sse(body):
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def test_stream_persists_only_completed_assistant(tmp_path, monkeypatch):
    import web_ui.main as api_main
    from fastapi.testclient import TestClient

    api_main.conversation_store = ConversationStore(tmp_path / "api.db")
    client = TestClient(api_main.app)
    conversation_id = client.post("/api/conversations", json={}).json()["id"]

    monkeypatch.setattr(api_main, "get_rag_service", lambda: FakeService(fail=True))
    failed = client.post("/api/chat/stream", json={"question": "追问", "conversation_id": conversation_id, "request_id": "bad"})
    assert _decode_sse(failed.text)[-1]["type"] == "error"
    assert [item["role"] for item in api_main.conversation_store.list_messages(conversation_id)] == ["user"]

    monkeypatch.setattr(api_main, "get_rag_service", lambda: FakeService())
    good = client.post("/api/chat/stream", json={"question": "继续", "conversation_id": conversation_id, "request_id": "ok"})
    assert _decode_sse(good.text)[-1]["type"] == "done"
    roles = [item["role"] for item in api_main.conversation_store.list_messages(conversation_id)]
    assert roles == ["user", "user", "assistant"]

    replay = client.post("/api/chat/stream", json={"question": "继续", "conversation_id": conversation_id, "request_id": "ok"})
    assert _decode_sse(replay.text)[-1]["replayed"] is True
    assert len(api_main.conversation_store.list_messages(conversation_id)) == 3
