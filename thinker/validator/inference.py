from __future__ import annotations

import contextlib
import copy
import hashlib
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thinker.config import ThinkerConfig
from thinker.validator.safe_adapter import materialize_validated_adapter

logger = logging.getLogger(__name__)
_PINNED_REVISION_RE = re.compile(r"[0-9a-fA-F]{40,64}")


def _chat_template_kwargs(enable_thinking: bool) -> dict[str, bool]:
    return {"enable_thinking": bool(enable_thinking)}


def _hash_adapter_files(adapter_files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(adapter_files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(adapter_files[name])
        digest.update(b"\0")
    return digest.hexdigest()


def _greedy_sampling_params(sampling_params: Any) -> Any:
    params = copy.copy(sampling_params)
    if hasattr(params, "temperature"):
        params.temperature = 0.0
    return params


def _make_lora_request(
    lora_request_cls: Any,
    *,
    lora_name: str,
    lora_int_id: int,
    adapter_path: str,
) -> Any:
    """Build a vLLM LoRARequest across versions.

    vLLM has used both `lora_path` and `lora_local_path` for the local adapter
    path. Try the current spelling first, then the older spelling, then a
    positional fallback for versions whose constructor does not expose keywords.
    """
    attempts = (
        {"lora_name": lora_name, "lora_int_id": lora_int_id, "lora_path": adapter_path},
        {
            "lora_name": lora_name,
            "lora_int_id": lora_int_id,
            "lora_local_path": adapter_path,
        },
    )
    last_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return lora_request_cls(**kwargs)
        except TypeError as exc:
            last_error = exc
    try:
        return lora_request_cls(lora_name, lora_int_id, adapter_path)
    except TypeError:
        if last_error is not None:
            raise last_error
        raise


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    completion: str
    token_count: int


class BaseModelServer:
    def __init__(
        self,
        base_model_path: str,
        base_model_revision: str | None = None,
        max_loras: int = 4,
        max_lora_rank: int = 128,
        security_policy: ThinkerConfig | None = None,
        show_progress: bool = False,
        **engine_kwargs: Any,
    ):
        if engine_kwargs.get("trust_remote_code") not in (None, False):
            raise ValueError("trust_remote_code must remain false")
        if engine_kwargs.get("load_format") not in (None, "safetensors"):
            raise ValueError("the base model must use safetensors")
        if "revision" in engine_kwargs:
            raise ValueError("pass base_model_revision explicitly; revision cannot be overridden")
        if not Path(base_model_path).is_dir():
            if (
                not isinstance(base_model_revision, str)
                or _PINNED_REVISION_RE.fullmatch(base_model_revision) is None
            ):
                raise ValueError(
                    "remote base models require an immutable 40-64 character commit revision"
                )
            engine_kwargs["revision"] = base_model_revision
        engine_kwargs["trust_remote_code"] = False
        engine_kwargs["load_format"] = "safetensors"

        from vllm import LLM

        self._llm = LLM(
            model=base_model_path,
            enable_lora=True,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            **engine_kwargs,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._next_lora_id = 1
        self._lora_ids: dict[str, int] = {}
        self._show_progress = show_progress
        self._security_policy = security_policy or ThinkerConfig(
            base_model_repo=base_model_path,
            base_model_revision=base_model_revision or "local",
            max_lora_rank=max_lora_rank,
        )

    def close(self) -> None:
        for candidate in (
            self._llm,
            getattr(self._llm, "llm_engine", None),
            getattr(self._llm, "engine", None),
        ):
            shutdown = getattr(candidate, "shutdown", None)
            if callable(shutdown):
                shutdown()
                return

    @contextmanager
    def _materialize_adapter(self, adapter_files: dict[str, bytes]):
        with materialize_validated_adapter(adapter_files, self._security_policy) as (
            tmpdir,
            _validated,
        ):
            yield tmpdir

    def _lora_id_for(self, adapter_hash: str) -> int:
        """Stable per-adapter-content LoRA id, reused across calls so vLLM's
        internal LoRA slot cache (bounded by `max_loras`) can recognize an
        adapter it already has resident instead of treating every call as a
        brand-new adapter -- this matters because the same miner's adapter is
        typically evaluated multiple times per epoch (qualification, then
        full eval).
        """
        lora_id = self._lora_ids.get(adapter_hash)
        if lora_id is None:
            lora_id = self._next_lora_id
            self._next_lora_id += 1
            self._lora_ids[adapter_hash] = lora_id
        return lora_id

    @staticmethod
    def _collect_results(prompts: list[str], outputs: list[Any]) -> list[GenerationResult]:
        results = []
        for prompt, output in zip(prompts, outputs):
            completion = output.outputs[0].text
            token_count = len(output.outputs[0].token_ids)
            results.append(
                GenerationResult(prompt=prompt, completion=completion, token_count=token_count)
            )
        return results

    @staticmethod
    def _collect_sample_sets(
        prompts: list[str], outputs: list[Any]
    ) -> list[list[GenerationResult]]:
        return [
            [
                GenerationResult(
                    prompt=prompt,
                    completion=candidate.text,
                    token_count=len(candidate.token_ids),
                )
                for candidate in output.outputs
            ]
            for prompt, output in zip(prompts, outputs)
        ]

    @staticmethod
    def _as_chat_messages(
        prompts: list[str], *, system_prompt: str | None = None
    ) -> list[list[dict[str, str]]]:
        if system_prompt:
            return [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                for prompt in prompts
            ]
        return [[{"role": "user", "content": prompt}] for prompt in prompts]

    def _chat(
        self,
        prompts: list[str],
        sampling_params: Any,
        *,
        enable_thinking: bool = True,
        system_prompt: str | None = None,
        **kwargs: Any,
    ):
        return self._llm.chat(
            self._as_chat_messages(prompts, system_prompt=system_prompt),
            sampling_params,
            chat_template_kwargs=_chat_template_kwargs(enable_thinking),
            use_tqdm=self._show_progress,
            **kwargs,
        )

    @contextmanager
    def suppress_progress(self):
        """Temporarily hide vLLM's per-call bars behind a caller-owned bar."""
        previous = self._show_progress
        self._show_progress = False
        try:
            yield
        finally:
            self._show_progress = previous

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _normalize_token_ids(self, value: Any) -> list[int]:
        if isinstance(value, dict):
            value = value.get("input_ids", [])
        if isinstance(value, str):
            value = self._tokenizer.encode(value)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], list)
        ):
            value = value[0]
        token_ids = list(value)
        if not all(isinstance(token_id, int) for token_id in token_ids):
            bad = next(
                token_id for token_id in token_ids if not isinstance(token_id, int)
            )
            bad_type = type(bad).__name__
            raise TypeError(
                "chat template produced non-integer token ids for vLLM "
                f"generation ({bad_type}); tokenizer must return integer ids"
            )
        return token_ids

    def _chat_token_ids(
        self,
        messages_list: list[list[dict[str, str]]],
        *,
        enable_thinking: bool = True,
        continue_final_message: bool = False,
    ) -> list[list[int]]:
        token_ids_list: list[list[int]] = []
        for messages in messages_list:
            add_generation_prompt = not continue_final_message
            try:
                kwargs = _chat_template_kwargs(enable_thinking)
                if continue_final_message:
                    kwargs["continue_final_message"] = True
                tokenized = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                    **kwargs,
                )
            except TypeError:
                if continue_final_message:
                    raise RuntimeError(
                        "tokenizer does not support continuing the final assistant "
                        "message; cannot inject retrieval output as an in-progress "
                        "tool result"
                    ) from None
                tokenized = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                tokenized = self._tokenizer.encode(messages[0]["content"])
            try:
                token_ids = self._normalize_token_ids(tokenized)
            except TypeError:
                try:
                    kwargs = _chat_template_kwargs(enable_thinking)
                    if continue_final_message:
                        kwargs["continue_final_message"] = True
                    rendered = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        **kwargs,
                    )
                except TypeError:
                    if continue_final_message:
                        raise RuntimeError(
                            "tokenizer does not support continuing the final assistant "
                            "message; cannot inject retrieval output as an in-progress "
                            "tool result"
                        ) from None
                    rendered = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                    )
                if isinstance(rendered, (list, tuple)) and all(
                    isinstance(chunk, str) for chunk in rendered
                ):
                    rendered = "".join(rendered)
                if not isinstance(rendered, str):
                    raise TypeError(
                        "chat template could not be rendered as text for vLLM generation"
                    )
                token_ids = self._normalize_token_ids(
                    self._tokenizer.encode(rendered)
                )
            token_ids_list.append(token_ids)
        return token_ids_list

    def count_chat_tokens(
        self,
        prompts: list[str],
        *,
        enable_thinking: bool = True,
        system_prompt: str | None = None,
    ) -> list[int]:
        messages_list = self._as_chat_messages(prompts, system_prompt=system_prompt)
        return [
            len(ids)
            for ids in self._chat_token_ids(messages_list, enable_thinking=enable_thinking)
        ]

    def count_message_tokens(
        self,
        messages_list: list[list[dict[str, str]]],
        *,
        enable_thinking: bool = True,
        continue_final_message: bool = False,
    ) -> list[int]:
        return [
            len(ids)
            for ids in self._chat_token_ids(
                messages_list,
                enable_thinking=enable_thinking,
                continue_final_message=continue_final_message,
            )
        ]

    def generate_original(
        self,
        prompts: list[str],
        sampling_params: Any,
        *,
        enable_thinking: bool = True,
        system_prompt: str | None = None,
    ) -> list[GenerationResult]:
        outputs = self._chat(
            prompts,
            sampling_params,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        )
        return self._collect_results(prompts, outputs)

    def generate_original_samples(
        self,
        prompts: list[str],
        sampling_params: Any,
        *,
        enable_thinking: bool = False,
        system_prompt: str | None = None,
    ) -> list[list[GenerationResult]]:
        outputs = self._chat(
            prompts,
            sampling_params,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        )
        return self._collect_sample_sets(prompts, outputs)

    def generate_for_miner(
        self,
        miner_uid: str,
        adapter_files: dict[str, bytes],
        prompts: list[str],
        sampling_params: Any,
        *,
        enable_thinking: bool = True,
        system_prompt: str | None = None,
    ) -> list[GenerationResult]:
        from vllm.lora.request import LoRARequest

        with self._materialize_adapter(adapter_files) as adapter_path:
            lora_request = _make_lora_request(
                LoRARequest,
                lora_name=f"miner-{miner_uid}",
                lora_int_id=self._lora_id_for(_hash_adapter_files(adapter_files)),
                adapter_path=adapter_path,
            )
            outputs = self._chat(
                prompts,
                sampling_params,
                enable_thinking=enable_thinking,
                system_prompt=system_prompt,
                lora_request=lora_request,
            )

        return self._collect_results(prompts, outputs)

    def generate_for_miner_messages(
        self,
        miner_uid: str,
        adapter_files: dict[str, bytes],
        messages_list: list[list[dict[str, str]]],
        sampling_params: Any,
        *,
        enable_thinking: bool = True,
    ) -> list[GenerationResult]:
        from vllm.lora.request import LoRARequest

        prompts = [
            next(
                (
                    str(message.get("content", ""))
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )
            for messages in messages_list
        ]
        with self._materialize_adapter(adapter_files) as adapter_path:
            lora_request = _make_lora_request(
                LoRARequest,
                lora_name=f"miner-{miner_uid}",
                lora_int_id=self._lora_id_for(_hash_adapter_files(adapter_files)),
                adapter_path=adapter_path,
            )
            outputs = self._llm.chat(
                messages_list,
                sampling_params,
                chat_template_kwargs=_chat_template_kwargs(enable_thinking),
                use_tqdm=self._show_progress,
                lora_request=lora_request,
            )

        return self._collect_results(prompts, outputs)

    def generate_original_messages_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        sampling_params_list: list[Any],
        *,
        enable_thinking: bool = True,
        continue_final_message: bool = False,
    ) -> list[GenerationResult]:
        if len(messages_list) != len(sampling_params_list):
            raise ValueError("messages_list and sampling_params_list must align")
        prompts = [
            next(
                (
                    str(message.get("content", ""))
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )
            for messages in messages_list
        ]
        token_ids_list = self._chat_token_ids(
            messages_list,
            enable_thinking=enable_thinking,
            continue_final_message=continue_final_message,
        )
        outputs = self._llm.generate(
            [{"prompt_token_ids": ids} for ids in token_ids_list],
            sampling_params_list,
            use_tqdm=self._show_progress,
        )
        if len(outputs) != len(messages_list):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} output(s) for {len(messages_list)} "
                "original message request(s)"
            )
        return self._collect_results(prompts, outputs)

    def generate_for_miners_messages_batch(
        self,
        requests: list[tuple[str, dict[str, bytes], list[dict[str, str]]]],
        sampling_params_list: list[Any],
        *,
        enable_thinking: bool = True,
        continue_final_message: bool = False,
    ) -> list[GenerationResult]:
        if not requests:
            return []
        if len(requests) != len(sampling_params_list):
            raise ValueError("requests and sampling_params_list must align")

        from vllm.lora.request import LoRARequest

        messages_list = [messages for _miner_uid, _adapter_files, messages in requests]
        prompts = [
            next(
                (
                    str(message.get("content", ""))
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )
            for messages in messages_list
        ]
        token_ids_list = self._chat_token_ids(
            messages_list,
            enable_thinking=enable_thinking,
            continue_final_message=continue_final_message,
        )

        with contextlib.ExitStack() as stack:
            lora_request_by_hash: dict[str, Any] = {}
            lora_requests: list[Any] = []
            for miner_uid, adapter_files, _messages in requests:
                adapter_hash = _hash_adapter_files(adapter_files)
                lora_request = lora_request_by_hash.get(adapter_hash)
                if lora_request is None:
                    adapter_path = stack.enter_context(self._materialize_adapter(adapter_files))
                    lora_request = _make_lora_request(
                        LoRARequest,
                        lora_name=f"miner-{miner_uid}",
                        lora_int_id=self._lora_id_for(adapter_hash),
                        adapter_path=adapter_path,
                    )
                    lora_request_by_hash[adapter_hash] = lora_request
                lora_requests.append(lora_request)

            outputs = self._llm.generate(
                [{"prompt_token_ids": ids} for ids in token_ids_list],
                sampling_params_list,
                lora_request=lora_requests,
                use_tqdm=self._show_progress,
            )

        if len(outputs) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} output(s) for {len(requests)} batched "
                "cross-miner message request(s)"
            )
        return self._collect_results(prompts, outputs)

    def generate_for_miners_batch(
        self,
        requests: list[tuple[str, dict[str, bytes], str]],
        sampling_params_list: list[Any],
        *,
        enable_thinking: bool = True,
        system_prompt: str | None = None,
    ) -> list[GenerationResult]:
        """Batch generation across *multiple miners'* adapters in a single
        vLLM call. Each entry in `requests` is `(miner_uid, adapter_files,
        prompt)`; the matching LoRA is attached per-request via vLLM's
        per-prompt `lora_request` list, so distinct miners' adapters are
        served side by side in one batch instead of one sequential call per
        miner.

        This relies on `LLM.generate(...)` accepting a list of prompts paired
        with an aligned list of `lora_request` entries (vLLM's documented
        multi-LoRA offline-inference pattern). Callers (see
        `EpochLoop._generate_miners_math_completions_batch`) treat any exception
        from this method as a signal to fall back to sequential per-miner
        generation, so a vLLM-version mismatch degrades gracefully instead of
        corrupting results.
        """
        if not requests:
            return []
        if len(requests) != len(sampling_params_list):
            raise ValueError("requests and sampling_params_list must align")

        from vllm.lora.request import LoRARequest

        prompts = [prompt for _miner_uid, _adapter_files, prompt in requests]
        messages_list = self._as_chat_messages(prompts, system_prompt=system_prompt)
        token_ids_list = self._chat_token_ids(messages_list, enable_thinking=enable_thinking)

        with contextlib.ExitStack() as stack:
            lora_request_by_hash: dict[str, Any] = {}
            lora_requests: list[Any] = []
            for miner_uid, adapter_files, _prompt in requests:
                adapter_hash = _hash_adapter_files(adapter_files)
                lora_request = lora_request_by_hash.get(adapter_hash)
                if lora_request is None:
                    adapter_path = stack.enter_context(self._materialize_adapter(adapter_files))
                    lora_request = _make_lora_request(
                        LoRARequest,
                        lora_name=f"miner-{miner_uid}",
                        lora_int_id=self._lora_id_for(adapter_hash),
                        adapter_path=adapter_path,
                    )
                    lora_request_by_hash[adapter_hash] = lora_request
                lora_requests.append(lora_request)

            outputs = self._llm.generate(
                [{"prompt_token_ids": ids} for ids in token_ids_list],
                sampling_params_list,
                lora_request=lora_requests,
                use_tqdm=self._show_progress,
            )

        if len(outputs) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} output(s) for {len(requests)} batched "
                "cross-miner request(s) -- aborting rather than risk misattributing results"
            )
        return self._collect_results(prompts, outputs)


class VllmInferenceBackend:
    def __init__(
        self,
        server: BaseModelServer,
        sampling_params: Any,
        *,
        max_total_tokens: int | None = None,
    ):
        self._server = server
        self._sampling_params = copy.copy(sampling_params)
        self._original_sampling_params = copy.copy(sampling_params)
        self._max_total_tokens = max_total_tokens

    @staticmethod
    def _as_tuples(results: list[GenerationResult]) -> list[tuple[str, int]]:
        return [(result.completion, result.token_count) for result in results]

    def _sampling_params_for_prompts(
        self,
        prompts: list[str],
        *,
        sampling_params: Any | None = None,
        max_new_tokens: int | None = None,
        enable_thinking: bool = True,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> Any:
        base_sampling_params = (
            self._sampling_params if sampling_params is None else sampling_params
        )
        if self._max_total_tokens is None or not prompts:
            params = copy.copy(base_sampling_params)
            if max_new_tokens is not None:
                configured_max = getattr(params, "max_tokens", None)
                params.max_tokens = (
                    max_new_tokens
                    if configured_max is None
                    else min(int(configured_max), max_new_tokens)
                )
            if stop:
                params.stop = list(stop)
                params.include_stop_str_in_output = True
            return params
        prompt_lengths = self._server.count_chat_tokens(
            prompts, enable_thinking=enable_thinking, system_prompt=system_prompt
        )
        remaining = [self._max_total_tokens - prompt_len for prompt_len in prompt_lengths]
        if any(tokens <= 0 for tokens in remaining):
            longest = max(prompt_lengths)
            raise ValueError(
                f"chat prompt has {longest} token(s), exceeding max_total_tokens="
                f"{self._max_total_tokens}"
            )
        max_tokens = min(remaining)
        configured_max = getattr(base_sampling_params, "max_tokens", None)
        if configured_max is not None:
            max_tokens = min(max_tokens, int(configured_max))
        if max_new_tokens is not None:
            max_tokens = min(max_tokens, max_new_tokens)
        params = copy.copy(base_sampling_params)
        params.max_tokens = max_tokens
        if stop:
            params.stop = list(stop)
            params.include_stop_str_in_output = True
        return params

    def _sampling_params_for_messages(
        self,
        messages: list[dict[str, str]],
        *,
        sampling_params: Any | None = None,
        max_new_tokens: int | None = None,
        enable_thinking: bool = True,
        stop: list[str] | None = None,
        continue_final_message: bool = False,
    ) -> Any:
        base_sampling_params = (
            self._sampling_params if sampling_params is None else sampling_params
        )
        if self._max_total_tokens is None:
            params = copy.copy(base_sampling_params)
            if max_new_tokens is not None:
                configured_max = getattr(params, "max_tokens", None)
                params.max_tokens = (
                    max_new_tokens
                    if configured_max is None
                    else min(int(configured_max), max_new_tokens)
                )
            if stop:
                params.stop = list(stop)
                params.include_stop_str_in_output = True
            return params
        prompt_len = self._server.count_message_tokens(
            [messages],
            enable_thinking=enable_thinking,
            continue_final_message=continue_final_message,
        )[0]
        remaining = self._max_total_tokens - prompt_len
        if remaining <= 0:
            raise ValueError(
                f"chat history has {prompt_len} token(s), exceeding max_total_tokens="
                f"{self._max_total_tokens}"
            )
        max_tokens = remaining
        configured_max = getattr(base_sampling_params, "max_tokens", None)
        if configured_max is not None:
            max_tokens = min(max_tokens, int(configured_max))
        if max_new_tokens is not None:
            max_tokens = min(max_tokens, max_new_tokens)
        params = copy.copy(base_sampling_params)
        params.max_tokens = max_tokens
        if stop:
            params.stop = list(stop)
            params.include_stop_str_in_output = True
        return params

    def generate_original(self, prompts: list[str]) -> list[tuple[str, int]]:
        return self.generate_original_limited(
            prompts, max_new_tokens=None, enable_thinking=True
        )

    def generate_original_limited(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int | None,
        enable_thinking: bool = True,
        system_prompt: str | None = None,
        stop: list[str] | None = None,
    ) -> list[tuple[str, int]]:
        return self._as_tuples(
            self._server.generate_original(
                prompts,
                self._sampling_params_for_prompts(
                    prompts,
                    sampling_params=self._original_sampling_params,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    system_prompt=system_prompt,
                    stop=stop,
                ),
                enable_thinking=enable_thinking,
                system_prompt=system_prompt,
            )
        )

    def generate_original_greedy_limited(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int | None,
        enable_thinking: bool = True,
        system_prompt: str | None = None,
        stop: list[str] | None = None,
    ) -> list[tuple[str, int]]:
        """Generate deterministic original-model output for dataset construction."""
        return self._as_tuples(
            self._server.generate_original(
                prompts,
                self._sampling_params_for_prompts(
                    prompts,
                    sampling_params=_greedy_sampling_params(
                        self._original_sampling_params
                    ),
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    system_prompt=system_prompt,
                    stop=stop,
                ),
                enable_thinking=enable_thinking,
                system_prompt=system_prompt,
            )
        )

    def generate_original_samples(
        self,
        prompts: list[str],
        *,
        num_samples: int,
        max_new_tokens: int | None,
        temperature: float,
        top_p: float,
        enable_thinking: bool = False,
    ) -> list[list[tuple[str, int]]]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("sampled generation requires temperature > 0")
        params = self._sampling_params_for_prompts(
            prompts,
            sampling_params=self._original_sampling_params,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
        params.temperature = float(temperature)
        params.top_p = float(top_p)
        params.n = int(num_samples)
        return [
            self._as_tuples(sample_set)
            for sample_set in self._server.generate_original_samples(
                prompts,
                params,
                enable_thinking=enable_thinking,
            )
        ]

    def generate(
        self, miner_id: str, adapter_files: dict[str, bytes], prompts: list[str]
    ) -> list[tuple[str, int]]:
        return self.generate_limited(
            miner_id, adapter_files, prompts, max_new_tokens=None
        )

    def generate_limited(
        self,
        miner_id: str,
        adapter_files: dict[str, bytes],
        prompts: list[str],
        *,
        max_new_tokens: int | None,
        enable_thinking: bool = True,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[tuple[str, int]]:
        return self._as_tuples(
            self._server.generate_for_miner(
                miner_id,
                adapter_files,
                prompts,
                self._sampling_params_for_prompts(
                    prompts,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    stop=stop,
                    system_prompt=system_prompt,
                ),
                enable_thinking=enable_thinking,
                system_prompt=system_prompt,
            )
        )

    def generate_chat(
        self,
        miner_id: str,
        adapter_files: dict[str, bytes],
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
        enable_thinking: bool = True,
        stop: list[str] | None = None,
    ) -> tuple[str, int]:
        results = self._server.generate_for_miner_messages(
            miner_id,
            adapter_files,
            [messages],
            self._sampling_params_for_messages(
                messages,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                stop=stop,
            ),
            enable_thinking=enable_thinking,
        )
        if len(results) != 1:
            raise RuntimeError(
                f"vLLM returned {len(results)} output(s) for one chat request"
            )
        return results[0].completion, results[0].token_count

    def generate_original_messages_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        *,
        max_new_tokens_list: list[int | None] | None = None,
        enable_thinking: bool = True,
        stop: list[str] | None = None,
        continue_final_message: bool = False,
    ) -> list[tuple[str, int]]:
        if not messages_list:
            return []
        if max_new_tokens_list is None:
            max_new_tokens_list = [None] * len(messages_list)
        if len(max_new_tokens_list) != len(messages_list):
            raise ValueError("max_new_tokens_list must align with messages_list")
        sampling_params_list = [
            self._sampling_params_for_messages(
                messages,
                sampling_params=self._original_sampling_params,
                max_new_tokens=budget,
                enable_thinking=enable_thinking,
                stop=stop,
                continue_final_message=continue_final_message,
            )
            for messages, budget in zip(messages_list, max_new_tokens_list)
        ]
        return self._as_tuples(
            self._server.generate_original_messages_batch(
                messages_list,
                sampling_params_list,
                enable_thinking=enable_thinking,
                continue_final_message=continue_final_message,
            )
        )

    def generate_for_miners_messages_batch(
        self,
        requests: list[tuple[str, dict[str, bytes], list[dict[str, str]]]],
        *,
        max_new_tokens_list: list[int | None] | None = None,
        enable_thinking: bool = True,
        stop: list[str] | None = None,
        continue_final_message: bool = False,
    ) -> list[tuple[str, int]]:
        if not requests:
            return []
        if max_new_tokens_list is None:
            max_new_tokens_list = [None] * len(requests)
        if len(max_new_tokens_list) != len(requests):
            raise ValueError("max_new_tokens_list must align with requests")

        sampling_params_list = [
            self._sampling_params_for_messages(
                messages,
                max_new_tokens=budget,
                enable_thinking=enable_thinking,
                stop=stop,
                continue_final_message=continue_final_message,
            )
            for (_miner_uid, _adapter_files, messages), budget in zip(
                requests, max_new_tokens_list
            )
        ]
        return self._as_tuples(
            self._server.generate_for_miners_messages_batch(
                requests,
                sampling_params_list,
                enable_thinking=enable_thinking,
                continue_final_message=continue_final_message,
            )
        )

    def generate_for_miners_batch(
        self,
        requests: list[tuple[str, dict[str, bytes], str]],
        *,
        max_new_tokens_list: list[int | None] | None = None,
        enable_thinking: bool = True,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[tuple[str, int]]:
        if not requests:
            return []
        if max_new_tokens_list is None:
            max_new_tokens_list = [None] * len(requests)
        if len(max_new_tokens_list) != len(requests):
            raise ValueError("max_new_tokens_list must align with requests")

        sampling_params_list = [
            self._sampling_params_for_prompts(
                [prompt],
                max_new_tokens=budget,
                enable_thinking=enable_thinking,
                stop=stop,
                system_prompt=system_prompt,
            )
            for (_miner_uid, _adapter_files, prompt), budget in zip(requests, max_new_tokens_list)
        ]
        return self._as_tuples(
            self._server.generate_for_miners_batch(
                requests,
                sampling_params_list,
                enable_thinking=enable_thinking,
                system_prompt=system_prompt,
            )
        )

    def count_tokens(self, text: str) -> int:
        return self._server.count_tokens(text)

    def suppress_progress(self):
        return self._server.suppress_progress()

    def close(self) -> None:
        self._server.close()
