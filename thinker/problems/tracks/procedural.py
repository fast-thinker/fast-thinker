from __future__ import annotations

import hashlib
import os

import reasoning_gym as rg

from thinker.problems.interface import Difficulty, register_track

ALL_GENERATOR_NAMES: tuple[str, ...] = (
    "gcd",
    "lcm",
    "gsm_symbolic",
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
    "gsm_symbolic",
    "polynomial_equations",
    "intermediate_integration",
    "simple_integration",
    "polynomial_multiplication",
    "power_function",
    "number_sequence",
    "calendar_arithmetic",
    "advanced_geometry",
)
GENERATOR_CONFIGS: dict[str, dict[str, object]] = {
    "polynomial_equations": {
        "min_terms": 3,
        "max_terms": 6,
        "min_value": 10,
        "max_value": 10_000,
        "min_degree": 2,
        "max_degree": 5,
    },
    "intermediate_integration": {
        "linear_upper_bound": 100,
        "min_linear_degree": 2,
        "max_linear_degree": 8,
        "outer_constant_max": 10,
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
    },
}

_EXACT_MATCH = 1.0


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
        _, ds, item = self._dataset_and_item(seed)
        try:
            score = ds.score_answer(answer=output, entry=item)
        except Exception:
            return False
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
