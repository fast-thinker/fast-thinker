from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from thinker.retrieval.bm25 import BM25RetrievalService, CorpusDocument, RetrievalHit


def _document_payload(document: CorpusDocument) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": document.doc_id,
        "title": document.title,
        "text": document.text,
        "contents": document.contents,
    }
    payload.update(document.metadata)
    return payload


def hits_payload(hits: list[RetrievalHit], *, return_scores: bool) -> list[dict[str, Any]]:
    if return_scores:
        return [
            {
                "document": _document_payload(hit.document),
                "score": hit.score,
                "rank": hit.rank,
            }
            for hit in hits
        ]
    return [_document_payload(hit.document) for hit in hits]


def retrieve_payload(
    retriever: BM25RetrievalService,
    *,
    queries: list[str],
    topk: int,
    return_scores: bool,
) -> dict[str, Any]:
    return {
        "result": [
            hits_payload(
                retriever.search(query, topk=topk),
                return_scores=return_scores,
            )
            for query in queries
        ]
    }


class RetrievalHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        retriever: BM25RetrievalService,
        *,
        default_topk: int = 50,
    ):
        super().__init__(server_address, RetrievalRequestHandler)
        self.retriever = retriever
        self.default_topk = default_topk


class RetrievalRequestHandler(BaseHTTPRequestHandler):
    server: RetrievalHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._write_json({"ok": True})
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/retrieve":
            self.send_error(404, "not found")
            return
        try:
            payload = self._read_json()
            queries = payload.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
                raise ValueError("queries must be a string list")
            topk = int(payload.get("topk") or self.server.default_topk)
            return_scores = bool(payload.get("return_scores", False))
            response = retrieve_payload(
                self.server.retriever,
                queries=queries,
                topk=topk,
                return_scores=return_scores,
            )
        except Exception as exc:
            self._write_json({"error": str(exc)}, status=400)
            return
        self._write_json(response)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _write_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@dataclass(frozen=True)
class RetrievalServiceHandle:
    server: RetrievalHTTPServer
    thread: threading.Thread
    url: str

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def start_retrieval_service(
    retriever: BM25RetrievalService,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    default_topk: int = 50,
) -> RetrievalServiceHandle:
    server = RetrievalHTTPServer((host, port), retriever, default_topk=default_topk)
    thread = threading.Thread(target=server.serve_forever, name="thinker-retrieval", daemon=True)
    thread.start()
    bound_host, bound_port = server.server_address
    return RetrievalServiceHandle(
        server=server,
        thread=thread,
        url=f"http://{bound_host}:{bound_port}",
    )


__all__ = [
    "RetrievalServiceHandle",
    "hits_payload",
    "retrieve_payload",
    "start_retrieval_service",
]
