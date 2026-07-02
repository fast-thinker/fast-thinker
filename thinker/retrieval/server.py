from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from thinker.retrieval.bm25 import BM25RetrievalService, CorpusDocument, RetrievalHit

logger = logging.getLogger(__name__)

DEFAULT_MAX_REQUEST_BYTES = 256 * 1024
DEFAULT_MAX_QUERIES = 32
DEFAULT_MAX_QUERY_CHARS = 4096
DEFAULT_MAX_TOPK = 100
DEFAULT_MAX_TOTAL_HITS = 1000
DEFAULT_MAX_CONCURRENT_REQUESTS = 2


class RequestError(ValueError):
    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.status = status


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
            hits_payload(hits, return_scores=return_scores)
            for hits in retriever.search_batch(queries, topk=topk)
        ]
    }


class RetrievalHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        retriever: BM25RetrievalService,
        *,
        default_topk: int = 50,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_queries: int = DEFAULT_MAX_QUERIES,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
        max_topk: int = DEFAULT_MAX_TOPK,
        max_total_hits: int = DEFAULT_MAX_TOTAL_HITS,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    ):
        if not 1 <= default_topk <= max_topk:
            raise ValueError("default_topk must be between 1 and max_topk")
        if min(
            max_request_bytes,
            max_queries,
            max_query_chars,
            max_topk,
            max_total_hits,
            max_concurrent_requests,
        ) <= 0:
            raise ValueError("retrieval server limits must be greater than zero")
        super().__init__(server_address, RetrievalRequestHandler)
        self.retriever = retriever
        self.default_topk = default_topk
        self.max_request_bytes = max_request_bytes
        self.max_queries = max_queries
        self.max_query_chars = max_query_chars
        self.max_topk = max_topk
        self.max_total_hits = max_total_hits
        self._worker_slots = threading.BoundedSemaphore(max_concurrent_requests + 1)
        self._retrieval_slots = threading.BoundedSemaphore(max_concurrent_requests)

    def process_request(self, request, client_address) -> None:
        # Keep one small overflow slot available to parse a request and return
        # a proper 503, while strictly bounding the number of handler threads.
        if not self._worker_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._worker_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_slots.release()


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
                raise RequestError("queries must be a string list")
            if len(queries) > self.server.max_queries:
                raise RequestError(f"at most {self.server.max_queries} queries are allowed")
            if any(len(query) > self.server.max_query_chars for query in queries):
                raise RequestError(
                    f"each query must be at most {self.server.max_query_chars} characters"
                )
            topk = int(payload.get("topk") or self.server.default_topk)
            if not 1 <= topk <= self.server.max_topk:
                raise RequestError(f"topk must be between 1 and {self.server.max_topk}")
            if len(queries) * topk > self.server.max_total_hits:
                raise RequestError(
                    f"queries * topk must not exceed {self.server.max_total_hits}"
                )
            return_scores = bool(payload.get("return_scores", False))
            if not self.server._retrieval_slots.acquire(blocking=False):
                raise RequestError("retrieval server is busy", status=503)
            try:
                response = retrieve_payload(
                    self.server.retriever,
                    queries=queries,
                    topk=topk,
                    return_scores=return_scores,
                )
            finally:
                self.server._retrieval_slots.release()
        except RequestError as exc:
            self._write_json({"error": str(exc)}, status=exc.status)
            return
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._write_json({"error": str(exc)}, status=400)
            return
        except Exception:
            logger.exception("BM25 retrieval request failed")
            self._write_json({"error": "retrieval failed"}, status=500)
            return
        self._write_json(response)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestError("invalid Content-Length") from exc
        if length < 0:
            raise RequestError("invalid Content-Length")
        if length > self.server.max_request_bytes:
            raise RequestError(
                f"request body must be at most {self.server.max_request_bytes} bytes",
                status=413,
            )
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
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    max_queries: int = DEFAULT_MAX_QUERIES,
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
    max_topk: int = DEFAULT_MAX_TOPK,
    max_total_hits: int = DEFAULT_MAX_TOTAL_HITS,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
) -> RetrievalServiceHandle:
    server = RetrievalHTTPServer(
        (host, port),
        retriever,
        default_topk=default_topk,
        max_request_bytes=max_request_bytes,
        max_queries=max_queries,
        max_query_chars=max_query_chars,
        max_topk=max_topk,
        max_total_hits=max_total_hits,
        max_concurrent_requests=max_concurrent_requests,
    )
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
