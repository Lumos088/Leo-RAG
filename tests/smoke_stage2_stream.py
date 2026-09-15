import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web_ui.api.rag_service import RAGService


async def main():
    service = RAGService()
    events = []
    answer = ""
    retrieval = []
    async for event in service.query_stream(
        "什么是并查集Union-Find", top_k=5, use_bm25=True, use_rerank=True
    ):
        suffix = f":{event.get('stage', '')}" if event["type"] == "progress" else ""
        events.append(event["type"] + suffix)
        answer = event.get("answer", answer)
        retrieval = event.get("retrieval", retrieval)
    print(f"events={events}")
    print(f"retrieval={len(retrieval)}")
    print(f"answer_chars={len(answer)}")
    print(f"done={bool(answer)}")


if __name__ == "__main__":
    asyncio.run(main())
