from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import queue

import reasoning_gym as rg

from thinker.problems.interface import (
    Difficulty,
    extract_final_boxed_answer,
    register_track,
)
from thinker.reward.verify import _apply_worker_limits

ALL_GENERATOR_NAMES: tuple[str, ...] = (
    "gcd",
    "lcm",
    "simple_equations",
    "polynomial_equations",
    "intermediate_integration",
    "simple_integration",
    "fraction_simplification",
    "prime_factorization",
    "count_primes",
    "complex_arithmetic",
    "decimal_chain_sum",
    "chain_sum",
    "polynomial_multiplication",
    "power_function",
    "base_conversion",
    "bitwise_arithmetic",
    "products",
    "number_sequence",
    "calendar_arithmetic",
    "advanced_geometry",
    "simple_geometry",
)
DEFAULT_GENERATOR_NAMES: tuple[str, ...] = (
    "polynomial_equations",
    "intermediate_integration",
    "advanced_geometry",
)
GENERATOR_CONFIGS: dict[str, dict[str, object]] = {
    "polynomial_equations": {
        "min_terms": 4,
        "max_terms": 6,
        "min_value": 10,
        "max_value": 10_000,
        "min_degree": 3,
        "max_degree": 5,
    },
    "intermediate_integration": {
        "problem_types": (
            "polynomial_exp_trig",
            "cyclic",
            "repeated_parts",
        ),
        "problem_type_weights": (1.0, 1.0, 1.0),
        "linear_upper_bound": 100,
        "min_linear_degree": 2,
        "max_linear_degree": 8,
        "outer_constant_max": 10,
        "min_poly_degree": 3,
        "max_poly_degree": 8,
    },
    "simple_integration": {
        "min_terms": 3,
        "max_terms": 8,
        "max_degree": 20,
        "max_bounds": 100,
    },
    "polynomial_multiplication": {
        "min_terms": 3,
        "max_terms": 6,
        "min_value": 10,
        "max_value": 10_000,
        "max_degree": 6,
        "max_polynomials": 4,
    },
    "power_function": {
        "min_base": -100_000.0,
        "max_base": 100_000.0,
        "max_exponent": 20,
    },
    "number_sequence": {
        "min_terms": 6,
        "max_terms": 12,
        "min_value": -10_000,
        "max_value": 10_000,
        "max_complexity": 5,
    },
    "prime_factorization": {
        "min_value": 10_000,
        "max_value": 100_000_000,
    },
    "gcd": {
        "min_numbers": 3,
        "max_numbers": 5,
        "min_value": 10_000,
        "max_value": 10_000_000,
    },
    "lcm": {
        "min_numbers": 3,
        "max_numbers": 5,
        "min_value": 100,
        "max_value": 100_000,
    },
    "base_conversion": {
        "min_base": 2,
        "max_base": 36,
        "min_value": 10_000,
        "max_value": 1_000_000_000,
    },
    "calendar_arithmetic": {
        "offset_upper_bound": 10_000,
        "leap_year_range": 10_000,
    },
    "advanced_geometry": {
        "min_coord": -1_000,
        "max_coord": 1_000,
        "task_types": ("orthocenter", "incircle_radius"),
    },
}

_EXACT_MATCH = 1.0
_VERIFY_TIMEOUT_S = 5.0
logger = logging.getLogger(__name__)


def _generator_names_from_env() -> tuple[str, ...]:
    raw = os.environ.get("THINKER_PROCEDURAL_GENERATORS")
    if raw is None:
        return DEFAULT_GENERATOR_NAMES
    selected = tuple(name.strip() for name in raw.split(",") if name.strip())
    unknown = sorted(set(selected) - set(ALL_GENERATOR_NAMES))
    if unknown:
        raise ValueError(
            "unknown THINKER_PROCEDURAL_GENERATORS entries: "
            f"{unknown}; known generators: {sorted(ALL_GENERATOR_NAMES)}"
        )
    if not selected:
        raise ValueError("THINKER_PROCEDURAL_GENERATORS cannot be empty")
    return selected


def _int_seed(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _generator_config(name: str, seed_value: int) -> dict[str, object]:
    config = dict(GENERATOR_CONFIGS.get(name, {}))
    if name == "calendar_arithmetic":
        config["year"] = 1_800 + seed_value % 10_000
    return config


def _score_answer_worker(
    seed: str,
    generator_names: tuple[str, ...],
    answer: str,
    response_queue: mp.Queue,
) -> None:
    _apply_worker_limits()
    try:
        seed_value = _int_seed(seed)
        name = generator_names[seed_value % len(generator_names)]
        config = _generator_config(name, seed_value)
        ds = rg.create_dataset(name, size=1, seed=seed_value, **config)
        response_queue.put(("ok", ds.score_answer(answer=answer, entry=ds[0])))
    except Exception as exc:
        response_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _score_answer_with_timeout(
    seed: str,
    generator_names: tuple[str, ...],
    answer: str,
    *,
    timeout: float = _VERIFY_TIMEOUT_S,
) -> float | None:
    context = mp.get_context("spawn")
    response_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_score_answer_worker,
        args=(seed, generator_names, answer, response_queue),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        response_queue.close()
        response_queue.join_thread()
        logger.warning(
            "reasoning-gym scorer exceeded %ss timeout; treating as unverified",
            timeout,
        )
        return None

    try:
        status, payload = response_queue.get_nowait()
    except queue.Empty:
        logger.warning(
            "reasoning-gym scorer exited without a result; treating as unverified"
        )
        return None
    finally:
        response_queue.close()
        response_queue.join_thread()

    if status != "ok":
        logger.debug("reasoning-gym scorer failed: %s", payload)
        return None
    return float(payload)


class ProceduralTrack:
    track = "procedural"

    def __init__(self, generator_names: tuple[str, ...] | None = None):
        if generator_names is None:
            generator_names = _generator_names_from_env()
        self._generator_names = generator_names

    def _pick_generator(self, seed: str) -> str:
        idx = _int_seed(seed) % len(self._generator_names)
        return self._generator_names[idx]

    def _dataset_and_item(self, seed: str):
        name = self._pick_generator(seed)
        seed_value = _int_seed(seed)
        config = _generator_config(name, seed_value)
        ds = rg.create_dataset(name, size=1, seed=seed_value, **config)
        return name, ds, ds[0]

    def render(self, seed: str) -> str:
        _, _, item = self._dataset_and_item(seed)
        return item["question"]

    def verify(self, seed: str, output: str) -> bool:
        answer = extract_final_boxed_answer(output)
        if answer is None:
            return False
        score = _score_answer_with_timeout(seed, self._generator_names, answer)
        return score == _EXACT_MATCH

    def difficulty(self, seed: str) -> Difficulty:
        name, _, item = self._dataset_and_item(seed)
        params = {"generator": name, **item.get("metadata", {})}
        return Difficulty(track=self.track, params=params)

    def min_tokens(self, seed: str) -> int:
        _, _, item = self._dataset_and_item(seed)
        return max(32, len(str(item["answer"])) // 2 + 16)


_track = ProceduralTrack()
register_track(_track)
