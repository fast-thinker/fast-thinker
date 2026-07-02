from __future__ import annotations

import unittest
from contextlib import nullcontext

from thinker.retrieval.bm25 import CorpusDocument, RetrievalHit
from thinker.validator.long_context_qa import (
    MINER_SYSTEM_PROMPT,
    LongContextQAConfig,
    LongContextQAEvaluator,
    LongContextQAInstance,
    parse_document_indices,
)


def _hit(rank: int, title: str, text: str) -> RetrievalHit:
    document = CorpusDocument(
        doc_id=str(rank),
        title=title,
        text=text,
        contents=f"{title}\n{text}",
    )
    return RetrievalHit(document=document, score=1.0 / rank, rank=rank)


class _Retriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits

    def search(self, _query: str, *, topk: int) -> list[RetrievalHit]:
        return self.hits[:topk]


class _Inference:
    def __init__(self) -> None:
        self.answer_prompts: list[str] = []

    def suppress_progress(self):
        return nullcontext()

    def generate_original_greedy_limited(self, prompts, **_kwargs):
        completions = []
        for prompt in prompts:
            if prompt.startswith("Use only the selected reference documents"):
                self.answer_prompts.append(prompt)
                answer = (
                    "Ada Lovelace"
                    if "Ada Lovelace is often called" in prompt
                    else "Charles Babbage"
                )
                completions.append((f"\\boxed{{{answer}}}", 4))
            else:
                completions.append(('{"equivalent": false}', 4))
        return completions

    def generate_original_limited(self, prompts, **kwargs):
        return self.generate_original_greedy_limited(prompts, **kwargs)


class LongContextEvidenceSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hits = [
            _hit(1, "Ada Lovelace", "Ada Lovelace is often called the first programmer."),
            _hit(2, "Charles Babbage", "Charles Babbage designed the Analytical Engine."),
        ]
        self.inference = _Inference()
        self.evaluator = LongContextQAEvaluator(
            retriever=_Retriever(self.hits),
            inference=self.inference,
            config=LongContextQAConfig(answer_context_topk=2),
        )
        self.instance = LongContextQAInstance(
            seed="seed",
            source_document=self.hits[0].document,
            seed_hits=tuple(self.hits),
            question="Who is often called the first programmer?",
            gold_answer="Ada Lovelace",
        )

    def test_prompt_requests_document_indices_in_box(self) -> None:
        self.assertIn("comma-separated Doc indices", MINER_SYSTEM_PROMPT)
        self.assertIn(r"\boxed{2,5}", MINER_SYSTEM_PROMPT)
        self.assertIn("Do not answer the question yourself", MINER_SYSTEM_PROMPT)

    def test_selected_documents_are_given_to_original_model(self) -> None:
        answers, selections = self.evaluator._resolve_evidence_selections(
            [self.instance],
            [r"\boxed{1}"],
            [7],
            ["first programmer"],
            source_label="test-miner",
        )

        self.assertEqual(selections, [(1,)])
        self.assertTrue(answers[0].verified)
        self.assertEqual(answers[0].text, "1")
        self.assertEqual(len(self.inference.answer_prompts), 1)
        self.assertIn("Ada Lovelace", self.inference.answer_prompts[0])
        self.assertNotIn("Charles Babbage designed", self.inference.answer_prompts[0])

    def test_wrong_selected_document_fails_answer_verification(self) -> None:
        answers, selections = self.evaluator._resolve_evidence_selections(
            [self.instance],
            [r"\boxed{2}"],
            [7],
            ["first programmer"],
            source_label="test-miner",
        )

        self.assertEqual(selections, [(2,)])
        self.assertFalse(answers[0].verified)

    def test_invalid_selection_is_rejected_without_answer_generation(self) -> None:
        answers, selections = self.evaluator._resolve_evidence_selections(
            [self.instance],
            [r"\boxed{3}"],
            [7],
            ["first programmer"],
            source_label="test-miner",
        )

        self.assertEqual(selections, [()])
        self.assertFalse(answers[0].verified)
        self.assertEqual(self.inference.answer_prompts, [])

    def test_document_index_parser_enforces_bounds_and_limit(self) -> None:
        self.assertEqual(
            parse_document_indices(r"\boxed{2,1,2}", max_index=2, max_selected=2),
            (2, 1),
        )
        self.assertIsNone(
            parse_document_indices(r"\boxed{3}", max_index=2, max_selected=2)
        )
        self.assertIsNone(
            parse_document_indices(r"\boxed{1,2}", max_index=2, max_selected=1)
        )


if __name__ == "__main__":
    unittest.main()
