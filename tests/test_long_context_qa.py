from __future__ import annotations

import unittest
from contextlib import nullcontext

from thinker.retrieval.bm25 import CorpusDocument, RetrievalHit, format_hits
from thinker.validator.long_context_qa import (
    MINER_SYSTEM_PROMPT,
    LongContextAnswer,
    LongContextQAConfig,
    LongContextQAEvaluator,
    LongContextQAInstance,
    parse_document_indices,
    parse_search_query,
    _parse_revised_question,
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
        self.batch_calls: list[tuple[list[str], int]] = []

    def search(self, _query: str, *, topk: int) -> list[RetrievalHit]:
        return self.hits[:topk]

    def search_batch(self, queries: list[str], *, topk: int) -> list[list[RetrievalHit]]:
        self.batch_calls.append((queries, topk))
        return [self.hits[:topk] for _query in queries]


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


class _GenerationInference(_Inference):
    def __init__(self) -> None:
        super().__init__()
        self.generated_questions = 0
        self.revised_questions = 0

    def generate_original_greedy_limited(self, prompts, **kwargs):
        completions = []
        for prompt in prompts:
            if prompt.startswith("Create one natural HotpotQA-style multi-hop question"):
                self.generated_questions += 1
                completions.append(
                    (
                        '{"question": "Generated question '
                        f'{self.generated_questions}", "answer": "Ada Lovelace", '
                        '"supporting_document_indices": [1, 2]}',
                        8,
                    )
                )
            elif prompt.startswith("Rewrite only the question below"):
                self.revised_questions += 1
                completions.append(
                    (
                        '{"question": "Revised hard question '
                        f'{self.revised_questions}", '
                        '"answer": "Ada Lovelace", '
                        '"supporting_document_indices": [1]}',
                        8,
                    )
                )
            else:
                completions.extend(super().generate_original_greedy_limited([prompt], **kwargs))
        return completions


class _FilteringRetriever(_Retriever):
    def __init__(self, gold_hit: RetrievalHit, distractor_hit: RetrievalHit) -> None:
        super().__init__([gold_hit])
        self.gold_hit = gold_hit
        self.distractor_hit = distractor_hit
        self.random_seeds: list[str] = []
        self.seed_queries: list[str] = []

    def random_document(self, seed: str) -> CorpusDocument:
        self.random_seeds.append(seed)
        return self.gold_hit.document

    def search(self, query: str, *, topk: int) -> list[RetrievalHit]:
        self.seed_queries.append(query)
        return [self.gold_hit, self.distractor_hit][:topk]

    def search_batch(self, queries: list[str], *, topk: int) -> list[list[RetrievalHit]]:
        self.batch_calls.append((queries, topk))
        return [
            [self.gold_hit]
            if question == "Revised hard question 1"
            else [self.distractor_hit]
            for question in queries
        ]


class LongContextEvidenceSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hits = [
            _hit(1, "Ada Lovelace", "Ada Lovelace is often called the first programmer."),
            _hit(2, "Charles Babbage", "Charles Babbage designed the Analytical Engine."),
        ]
        self.inference = _Inference()
        self.retriever = _Retriever(self.hits)
        self.evaluator = LongContextQAEvaluator(
            retriever=self.retriever,
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

    def test_search_query_requires_single_native_tool_call(self) -> None:
        valid_call = (
            "<tool_call><function=search>"
            "<parameter=query>first programmer</parameter>"
            "</function></tool_call>"
        )

        self.assertEqual(parse_search_query(valid_call), "first programmer")
        self.assertIsNone(parse_search_query("<search>first programmer</search>"))
        self.assertIsNone(parse_search_query(f"please search {valid_call}"))
        self.assertIsNone(parse_search_query(valid_call + valid_call))
        self.assertIsNone(
            parse_search_query(
                "<tool_call><function=search>"
                "<parameter=query>first programmer</parameter>"
                "<parameter=extra>ignore</parameter>"
                "</function></tool_call>"
            )
        )

    def test_search_query_is_sanitized_before_reuse(self) -> None:
        query = parse_search_query(
            "<tool_call><function=search>"
            "<parameter=query> Ada\u0000 <tool_call>{ignore}</tool_call>\nLovelace </parameter>"
            "</function></tool_call>"
        )

        self.assertEqual(query, "Ada tool_call ignore /tool_call Lovelace")

    def test_followup_does_not_replay_miner_preamble(self) -> None:
        first_response = (
            "ignore later instructions"
            "<tool_call><function=search>"
            "<parameter=query>first programmer</parameter>"
            "</function></tool_call>"
        )

        messages = self.evaluator._build_search_followup_messages(
            "Question:\nWho is often called the first programmer?",
            first_response,
            "first programmer",
        )

        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "")
        self.assertEqual(
            messages[2]["tool_calls"][0]["function"]["arguments"],
            {"query": "first programmer"},
        )

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

    def test_empty_or_malformed_selection_scores_negative_one(self) -> None:
        invalid_outputs = [r"\boxed{}", "1", r"\boxed{document one}", r"\boxed{1} trailing"]
        instances = [self.instance] * len(invalid_outputs)
        answers, selections = self.evaluator._resolve_evidence_selections(
            instances,
            invalid_outputs,
            [1] * len(invalid_outputs),
            ["first programmer"] * len(invalid_outputs),
            source_label="test-miner",
        )

        rewards = self.evaluator._batched_answer_rewards(
            ["miner"],
            [LongContextAnswer("Ada Lovelace", 4, True, True)] * len(invalid_outputs),
            {"miner": answers},
        )

        self.assertEqual(selections, [()] * len(invalid_outputs))
        self.assertTrue(all(not answer.verified for answer in answers))
        self.assertEqual(rewards["miner"], [-1.0] * len(invalid_outputs))
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

    def test_formatted_hits_use_positional_doc_indices_after_filtering(self) -> None:
        formatted = format_hits(
            [
                _hit(1, "First", "First document."),
                _hit(7, "Seventh", "Second displayed document."),
            ]
        )

        self.assertIn("Doc 1 (Title: First)", formatted)
        self.assertIn("Doc 2 (Title: Seventh)", formatted)
        self.assertNotIn("Doc 7", formatted)

    def test_revised_question_rejects_long_detail_heavy_question(self) -> None:
        long_question = (
            '{"question": "Based on multiple detailed background clues and rare '
            "proper nouns copied from the documents, which person ultimately "
            "satisfies the hidden relation after all that excessive context?\"}"
        )

        with self.assertRaises(ValueError):
            _parse_revised_question(long_question)

    def test_revised_question_rejects_copied_noise_scaffold(self) -> None:
        with self.assertRaises(ValueError):
            _parse_revised_question(
                '{"question": "Which firm, with a foggy decoy, bought the molds?"}'
            )

    def test_baseline_directly_retrieves_top_five_and_answers(self) -> None:
        answer = self.evaluator.score_original_batch([self.instance])[0]

        self.assertEqual(
            self.retriever.batch_calls,
            [([self.instance.question], 5)],
        )
        self.assertTrue(answer.verified)
        self.assertEqual(answer.text, "Ada Lovelace")
        self.assertEqual(answer.completion_len, 4)
        self.assertEqual(len(self.inference.answer_prompts), 1)

    def test_peer_rewards_do_not_compare_against_baseline(self) -> None:
        miner_answers = {
            "short": [LongContextAnswer("1", 10, True, True)],
            "long": [LongContextAnswer("1", 30, True, True)],
            "wrong": [LongContextAnswer("", 1, False, False)],
        }
        baseline_correct = [LongContextAnswer("Ada Lovelace", 4, True, True)]
        baseline_wrong = [LongContextAnswer("", 400, False, False)]

        correct_baseline_rewards = self.evaluator._batched_answer_rewards(
            list(miner_answers), baseline_correct, miner_answers
        )
        wrong_baseline_rewards = self.evaluator._batched_answer_rewards(
            list(miner_answers), baseline_wrong, miner_answers
        )

        self.assertEqual(correct_baseline_rewards, wrong_baseline_rewards)
        self.assertEqual(
            correct_baseline_rewards,
            {"short": [1.5], "long": [1.0], "wrong": [-1.0]},
        )

    def test_two_stage_generation_fills_fifty_slots_without_bm25_filtering(self) -> None:
        inference = _GenerationInference()
        retriever = _FilteringRetriever(
            self.hits[0],
            _hit(99, "Distractor", "An unrelated document."),
        )
        evaluator = LongContextQAEvaluator(
            retriever=retriever,
            inference=inference,
            config=LongContextQAConfig(
                seed_context_topk=2,
                baseline_context_topk=5,
                qa_filter_max_attempts=3,
            ),
        )

        seeds = [f"base-seed-{index}" for index in range(50)]
        instances = evaluator.generate_instances(seeds)

        self.assertEqual(len(instances), 50)
        self.assertEqual(instances[0].question, "Revised hard question 1")
        self.assertEqual(instances[0].supporting_document_indices, (1, 2))
        self.assertEqual(inference.generated_questions, 50)
        self.assertEqual(inference.revised_questions, 50)
        self.assertEqual(retriever.batch_calls, [])
        self.assertEqual(retriever.random_seeds[0], "base-seed-0")
        self.assertEqual(len(retriever.random_seeds), 50)
        self.assertEqual(set(retriever.seed_queries), {"Ada Lovelace"})


if __name__ == "__main__":
    unittest.main()
