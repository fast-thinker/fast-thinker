from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from typing import Callable

from thinker.problems.interface import Difficulty, register_track

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}", re.DOTALL)
_INT_RE = re.compile(r"-?\d+")


def _int_seed(seed: str, salt: str = "") -> int:
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(seed: str) -> random.Random:
    return random.Random(_int_seed(seed, "constructive"))


def _last_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _candidate_ints(output: str) -> list[int]:
    content = _last_boxed(output) or output
    return [int(match.group(0)) for match in _INT_RE.finditer(content)]


def _coprime_pair(rng: random.Random, lo: int, hi: int) -> tuple[int, int]:
    while True:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if math.gcd(a, b) == 1:
            return a, b


def _count_inversions(values: list[int]) -> int:
    return sum(1 for i, left in enumerate(values) for right in values[i + 1:] if left > right)


def _permutation_with_inversions(n: int, k: int) -> list[int]:
    perm: list[int] = []
    remaining = k
    for value in range(n, 0, -1):
        pos = min(remaining, len(perm))
        perm.insert(pos, value)
        remaining -= pos
    return perm


@dataclass(frozen=True)
class ConstructiveInstance:
    family: str
    prompt: str
    solution: tuple[int, ...]
    checker: Callable[[list[int]], bool] = field(repr=False)
    params: dict[str, object] = field(default_factory=dict)
    min_tokens_hint: int = 160

    def get_solution(self) -> str:
        return ", ".join(str(x) for x in self.solution)


def _mod_inverse(rng: random.Random) -> ConstructiveInstance:
    modulus = rng.randint(41, 499)
    while True:
        a = rng.randint(2, modulus - 2)
        if math.gcd(a, modulus) == 1:
            break
    answer = pow(a, -1, modulus)
    prompt = (
        f"Find an integer x with 0 <= x < {modulus} such that "
        f"{a} * x leaves remainder 1 when divided by {modulus}. "
        "End with \\boxed{x}."
    )
    return ConstructiveInstance(
        family="mod_inverse",
        prompt=prompt,
        solution=(answer,),
        checker=lambda xs: bool(xs) and 0 <= xs[0] < modulus and (a * xs[0]) % modulus == 1,
        params={"problem_class": "mod_inverse", "a": a, "modulus": modulus},
        min_tokens_hint=96,
    )


def _linear_diophantine(rng: random.Random) -> ConstructiveInstance:
    a, b = _coprime_pair(rng, 12, 90)
    x0 = rng.randint(-25, 25)
    y0 = rng.randint(-25, 25)
    c = a * x0 + b * y0
    prompt = (
        f"Find integers x and y satisfying {a}x + {b}y = {c}. "
        "End with \\boxed{x,y}."
    )
    return ConstructiveInstance(
        family="linear_diophantine",
        prompt=prompt,
        solution=(x0, y0),
        checker=lambda xs: len(xs) >= 2 and a * xs[0] + b * xs[1] == c,
        params={"problem_class": "linear_diophantine", "a": a, "b": b, "c": c},
    )


def _crt(rng: random.Random) -> ConstructiveInstance:
    m1, m2 = _coprime_pair(rng, 5, 31)
    modulus = m1 * m2
    answer = rng.randint(0, modulus - 1)
    r1 = answer % m1
    r2 = answer % m2
    prompt = (
        f"Find the smallest nonnegative integer x such that x = {r1} mod {m1} "
        f"and x = {r2} mod {m2}. End with \\boxed{{x}}."
    )
    return ConstructiveInstance(
        family="crt",
        prompt=prompt,
        solution=(answer,),
        checker=lambda xs: bool(xs)
        and xs[0] == answer
        and xs[0] % m1 == r1
        and xs[0] % m2 == r2,
        params={"problem_class": "crt", "m1": m1, "m2": m2, "r1": r1, "r2": r2},
    )


def _permutation(rng: random.Random) -> ConstructiveInstance:
    n = rng.randint(5, 9)
    max_inv = n * (n - 1) // 2
    k = rng.randint(max(1, n - 3), max_inv - 1)
    answer = tuple(_permutation_with_inversions(n, k))
    prompt = (
        f"Construct a permutation of 1, 2, ..., {n} with exactly {k} inversions. "
        "End with \\boxed{p1,p2,...,pn}."
    )

    def checker(xs: list[int]) -> bool:
        candidate = xs[:n]
        return sorted(candidate) == list(range(1, n + 1)) and _count_inversions(candidate) == k

    return ConstructiveInstance(
        family="permutation_inversions",
        prompt=prompt,
        solution=answer,
        checker=checker,
        params={"problem_class": "permutation_inversions", "n": n, "inversions": k},
        min_tokens_hint=220,
    )


def _egyptian_fraction(rng: random.Random) -> ConstructiveInstance:
    n = rng.randint(5, 80)
    a = n + 1
    b = n * (n + 1)
    prompt = (
        f"Find positive integers a and b such that 1/a + 1/b = 1/{n}. "
        "End with \\boxed{a,b}."
    )

    def checker(xs: list[int]) -> bool:
        if len(xs) < 2 or xs[0] <= 0 or xs[1] <= 0:
            return False
        return n * (xs[0] + xs[1]) == xs[0] * xs[1]

    return ConstructiveInstance(
        family="egyptian_fraction",
        prompt=prompt,
        solution=(a, b),
        checker=checker,
        params={"problem_class": "egyptian_fraction", "denominator": n},
        min_tokens_hint=128,
    )


def _gcd_lcm(rng: random.Random) -> ConstructiveInstance:
    g = rng.randint(2, 40)
    u, v = _coprime_pair(rng, 2, 35)
    x = g * u
    y = g * v
    lcm = g * u * v
    prompt = (
        f"Find two positive integers x and y with gcd(x,y) = {g} and "
        f"lcm(x,y) = {lcm}. End with \\boxed{{x,y}}."
    )

    def checker(xs: list[int]) -> bool:
        if len(xs) < 2 or xs[0] <= 0 or xs[1] <= 0:
            return False
        return math.gcd(xs[0], xs[1]) == g and math.lcm(xs[0], xs[1]) == lcm

    return ConstructiveInstance(
        family="gcd_lcm",
        prompt=prompt,
        solution=(x, y),
        checker=checker,
        params={"problem_class": "gcd_lcm", "gcd": g, "lcm": lcm},
    )


def _pythagorean(rng: random.Random) -> ConstructiveInstance:
    m, n = _coprime_pair(rng, 3, 24)
    if n > m:
        m, n = n, m
    if (m - n) % 2 == 0:
        m += 1
    a = m * m - n * n
    b = 2 * m * n
    c = m * m + n * n
    prompt = (
        f"Find positive integers a and b such that a^2 + b^2 = {c}^2. "
        "End with \\boxed{a,b}."
    )

    def checker(xs: list[int]) -> bool:
        if len(xs) < 2 or xs[0] <= 0 or xs[1] <= 0:
            return False
        return xs[0] * xs[0] + xs[1] * xs[1] == c * c

    return ConstructiveInstance(
        family="pythagorean",
        prompt=prompt,
        solution=(a, b),
        checker=checker,
        params={"problem_class": "pythagorean", "hypotenuse": c},
    )


def _quadratic_residue(rng: random.Random) -> ConstructiveInstance:
    prime = rng.choice((101, 103, 107, 109, 113, 127, 131, 137, 139, 149))
    answer = rng.randint(2, prime - 2)
    residue = (answer * answer) % prime
    prompt = (
        f"Find an integer x with 0 <= x < {prime} such that x^2 = {residue} mod {prime}. "
        "End with \\boxed{x}."
    )
    return ConstructiveInstance(
        family="quadratic_residue",
        prompt=prompt,
        solution=(answer,),
        checker=lambda xs: bool(xs) and 0 <= xs[0] < prime and (xs[0] * xs[0]) % prime == residue,
        params={"problem_class": "quadratic_residue", "prime": prime, "residue": residue},
    )


_FAMILIES: tuple[Callable[[random.Random], ConstructiveInstance], ...] = (
    _mod_inverse,
    _linear_diophantine,
    _crt,
    _permutation,
    _egyptian_fraction,
    _gcd_lcm,
    _pythagorean,
    _quadratic_residue,
)


class ConstructiveTrack:
    track = "constructive"

    def _family_index(self, seed: str) -> int:
        return _int_seed(seed, "family") % len(_FAMILIES)

    def _instance(self, seed: str) -> ConstructiveInstance:
        rng = _rng(seed)
        return _FAMILIES[self._family_index(seed)](rng)

    def render(self, seed: str) -> str:
        return self._instance(seed).prompt

    def verify(self, seed: str, output: str) -> bool:
        try:
            return bool(self._instance(seed).checker(_candidate_ints(output)))
        except Exception:
            return False

    def difficulty(self, seed: str) -> Difficulty:
        inst = self._instance(seed)
        return Difficulty(track=self.track, params={"source": "thinker", **inst.params})

    def min_tokens(self, seed: str) -> int:
        return self._instance(seed).min_tokens_hint


_track = ConstructiveTrack()
register_track(_track)
