from __future__ import annotations

import hashlib

import reasoning_gym as rg

from thinker.problems.interface import Difficulty, register_track

GENERATOR_NAMES: tuple[str, ...] = (
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

_EXACT_MATCH = 1.0


def _int_seed(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class ProceduralTrack:
    track = "procedural"

    def __init__(self, generator_names: tuple[str, ...] = GENERATOR_NAMES):
        self._generator_names = generator_names

    def _pick_generator(self, seed: str) -> str:
        idx = _int_seed(seed) % len(self._generator_names)
        return self._generator_names[idx]

    def _dataset_and_item(self, seed: str):
        name = self._pick_generator(seed)
        ds = rg.create_dataset(name, size=1, seed=_int_seed(seed))
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
