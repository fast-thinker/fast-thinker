from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from thinker.retrieval.bm25 import BM25RetrievalService, CorpusDocument
from thinker.retrieval.server import retrieve_payload, start_retrieval_service


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def tokenize(self, queries: list[str], *, stopwords):
        self.calls.append(queries)
        return queries


class _Retriever:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def retrieve(self, queries: list[str], *, k: int):
        self.calls.append((queries, k))
        indexes = [[0, 1][:k] for _ in queries]
        scores = [[2.0, 1.0][:k] for _ in queries]
        return indexes, scores


def _service() -> BM25RetrievalService:
    service = BM25RetrievalService.__new__(BM25RetrievalService)
    service.documents = [
        CorpusDocument("1", "One", "first", "One\nfirst"),
        CorpusDocument("2", "Two", "second", "Two\nsecond"),
    ]
    service.stopwords = "en"
    service._bm25s = _Tokenizer()
    service._retriever = _Retriever()
    return service


class BM25BatchSearchTest(unittest.TestCase):
    def test_search_batch_uses_one_retriever_call_and_preserves_empty_queries(self) -> None:
        service = _service()

        results = service.search_batch(["alpha", "   ", "beta"], topk=2)

        self.assertEqual(
            [[hit.document.doc_id for hit in hits] for hits in results],
            [["1", "2"], [], ["1", "2"]],
        )
        self.assertEqual(service._bm25s.calls, [["alpha", "beta"]])
        self.assertEqual(service._retriever.calls, [(["alpha", "beta"], 2)])

    def test_retrieve_payload_uses_batch_search(self) -> None:
        service = _service()

        payload = retrieve_payload(
            service, queries=["alpha", "beta"], topk=1, return_scores=False
        )

        self.assertEqual(len(payload["result"]), 2)
        self.assertEqual(len(service._retriever.calls), 1)


class RetrievalServerLimitsTest(unittest.TestCase):
    def _post(self, url: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        request = Request(
            f"{url}/retrieve",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urlopen(request, timeout=2)
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())
        with response:
            return response.status, json.loads(response.read())

    def test_rejects_query_and_topk_limits(self) -> None:
        handle = start_retrieval_service(
            _service(),
            port=0,
            default_topk=1,
            max_queries=2,
            max_topk=2,
            max_total_hits=2,
        )
        try:
            status, _ = self._post(handle.url, {"queries": ["a", "b", "c"]})
            self.assertEqual(status, 400)
            status, _ = self._post(handle.url, {"queries": ["a"], "topk": 3})
            self.assertEqual(status, 400)
            status, _ = self._post(handle.url, {"queries": ["a", "b"], "topk": 2})
            self.assertEqual(status, 400)
        finally:
            handle.stop()

    def test_rejects_oversized_request_body(self) -> None:
        handle = start_retrieval_service(_service(), port=0, max_request_bytes=32)
        try:
            status, body = self._post(handle.url, {"queries": ["x" * 64]})
            self.assertEqual(status, 413)
            self.assertIn("request body", str(body["error"]))
        finally:
            handle.stop()

    def test_rejects_concurrent_work_instead_of_queuing_threads(self) -> None:
        service = _service()
        entered = threading.Event()
        release = threading.Event()
        original = service.search_batch

        def slow_search(queries, *, topk=10):
            entered.set()
            release.wait(timeout=2)
            return original(queries, topk=topk)

        service.search_batch = slow_search
        handle = start_retrieval_service(service, port=0, max_concurrent_requests=1)
        first_result: list[tuple[int, dict[str, object]]] = []
        first = threading.Thread(
            target=lambda: first_result.append(self._post(handle.url, {"queries": ["first"]}))
        )
        try:
            first.start()
            self.assertTrue(entered.wait(timeout=1))
            status, body = self._post(handle.url, {"queries": ["second"]})
            self.assertEqual(status, 503)
            self.assertIn("busy", str(body["error"]))
        finally:
            release.set()
            first.join(timeout=2)
            handle.stop()
        self.assertEqual(first_result[0][0], 200)


if __name__ == "__main__":
    unittest.main()
