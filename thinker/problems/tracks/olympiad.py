from __future__ import annotations

import functools
import hashlib
import math
import random
import re
from dataclasses import dataclass

from thinker.problems.interface import Difficulty, register_track


_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", re.DOTALL)
_INTEGER_RE = re.compile(r"[+-]?\d+")


def _int_seed(seed: str, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(seed: str, salt: str) -> random.Random:
    return random.Random(_int_seed(seed, salt))


@dataclass(frozen=True)
class ParallelChordsSpec:
    radius: int
    first_chord: int
    second_chord: int
    surface: int


@dataclass(frozen=True)
class CevianSpec:
    side_ab: int
    side_ac: int
    side_bc: int
    segment_bd: int
    surface: int


@dataclass(frozen=True)
class SymmetricRootsSpec:
    sum_roots: int
    sum_pairwise: int
    product: int
    power: int
    hidden_roots: tuple[int, int, int]
    surface: int


@dataclass(frozen=True)
class LatticePathsSpec:
    width: int
    height: int
    first_forbidden: tuple[int, int]
    second_forbidden: tuple[int, int]
    surface: int


@dataclass(frozen=True)
class ModularPowerSpec:
    base: int
    exponent_base: int
    exponent_power: int
    exponent_offset: int
    surface: int


@dataclass(frozen=True)
class RecurrenceSpec:
    first: int
    multiplier: int
    increment: int
    index: int
    modulus: int
    surface: int


@dataclass(frozen=True)
class ExactlyOneDivisorSpec:
    limit: int
    divisors: tuple[int, int, int]
    surface: int


OlympiadSpec = (
    ParallelChordsSpec
    | CevianSpec
    | SymmetricRootsSpec
    | LatticePathsSpec
    | ModularPowerSpec
    | RecurrenceSpec
    | ExactlyOneDivisorSpec
)


@dataclass(frozen=True)
class OlympiadInstance:
    family: str
    spec: OlympiadSpec
    prompt: str
    answer: int
    reasoning_steps: int

    def get_solution(self) -> str:
        return str(self.answer)


def _build_parallel_chords(rng: random.Random, surface: int) -> ParallelChordsSpec:
    systems = (
        (65, ((16, 63), (25, 60), (33, 56), (39, 52))),
        (85, ((13, 84), (36, 77), (40, 75), (51, 68))),
        (125, ((35, 120), (44, 117), (75, 100))),
    )
    radius, pairs = rng.choice(systems)
    scale = rng.randint(1, 60)
    first, second = rng.sample(list(pairs), 2)
    return ParallelChordsSpec(
        radius=radius * scale,
        first_chord=2 * first[0] * scale,
        second_chord=2 * second[0] * scale,
        surface=surface,
    )


def _build_cevian(rng: random.Random, surface: int) -> CevianSpec:
    systems = (
        (60, ((11, 61), (25, 65), (45, 75), (80, 100))),
        (84, ((13, 85), (35, 91), (63, 105))),
        (120, ((22, 122), (50, 130), (90, 150), (160, 200))),
    )
    altitude, pairs = rng.choice(systems)
    scale = rng.randint(1, 60)
    left, right = rng.sample(list(pairs), 2)
    bd, ab = left
    dc, ac = right
    return CevianSpec(
        side_ab=ab * scale,
        side_ac=ac * scale,
        side_bc=(bd + dc) * scale,
        segment_bd=bd * scale,
        surface=surface,
    )


def _build_symmetric_roots(rng: random.Random, surface: int) -> SymmetricRootsSpec:
    roots = tuple(sorted(rng.sample(range(3, 31), 3)))
    x, y, z = roots
    return SymmetricRootsSpec(
        sum_roots=x + y + z,
        sum_pairwise=x * y + x * z + y * z,
        product=x * y * z,
        power=rng.choice((4, 5)),
        hidden_roots=roots,
        surface=surface,
    )


def _build_lattice_paths(rng: random.Random, surface: int) -> LatticePathsSpec:
    width = rng.randint(11, 17)
    height = rng.randint(11, 17)
    px = rng.randint(2, width - 6)
    py = rng.randint(2, height - 6)
    qx = rng.randint(px + 2, width - 2)
    qy = rng.randint(py + 2, height - 2)
    return LatticePathsSpec(
        width=width,
        height=height,
        first_forbidden=(px, py),
        second_forbidden=(qx, qy),
        surface=surface,
    )


def _build_modular_power(rng: random.Random, surface: int) -> ModularPowerSpec:
    return ModularPowerSpec(
        base=rng.choice((3, 7, 11, 13, 17, 19, 23, 27)),
        exponent_base=rng.randint(3, 11),
        exponent_power=rng.randint(8, 16),
        exponent_offset=rng.randint(20, 500),
        surface=surface,
    )


def _build_recurrence(rng: random.Random, surface: int) -> RecurrenceSpec:
    return RecurrenceSpec(
        first=rng.randint(2, 200),
        multiplier=rng.randint(2, 9),
        increment=rng.randint(3, 80),
        index=rng.randint(150, 700),
        modulus=rng.choice((997, 1000, 1009, 2027)),
        surface=surface,
    )


def _build_exactly_one(rng: random.Random, surface: int) -> ExactlyOneDivisorSpec:
    divisor_sets = (
        (6, 10, 15),
        (8, 12, 18),
        (9, 14, 20),
        (10, 14, 21),
        (12, 15, 28),
    )
    return ExactlyOneDivisorSpec(
        limit=rng.randint(8_000, 40_000),
        divisors=rng.choice(divisor_sets),
        surface=surface,
    )


_BUILDERS = (
    ("parallel_chords", _build_parallel_chords, 5),
    ("cevian_length", _build_cevian, 6),
    ("symmetric_roots", _build_symmetric_roots, 7),
    ("lattice_paths", _build_lattice_paths, 8),
    ("modular_power", _build_modular_power, 7),
    ("linear_recurrence", _build_recurrence, 6),
    ("exactly_one_divisor", _build_exactly_one, 6),
)


def _distance_from_center(radius: int, chord: int) -> int:
    if chord % 2:
        raise ValueError("chord length must be even")
    squared = radius * radius - (chord // 2) ** 2
    distance = math.isqrt(squared)
    if distance * distance != squared:
        raise ValueError("chord does not have an integral center distance")
    return distance


def _solve_parallel_chords(spec: ParallelChordsSpec) -> int:
    return _distance_from_center(spec.radius, spec.first_chord) + _distance_from_center(
        spec.radius, spec.second_chord
    )


def _solve_cevian(spec: CevianSpec) -> int:
    m = spec.segment_bd
    n = spec.side_bc - m
    numerator = spec.side_ac**2 * m + spec.side_ab**2 * n
    if numerator % spec.side_bc:
        raise ValueError("Stewart numerator is not divisible by the opposite side")
    squared = numerator // spec.side_bc - m * n
    length = math.isqrt(squared)
    if length * length != squared:
        raise ValueError("cevian length is not integral")
    return length


def _solve_symmetric_roots(spec: SymmetricRootsSpec) -> int:
    e1, e2, e3 = spec.sum_roots, spec.sum_pairwise, spec.product
    powers = {0: 3, 1: e1}
    powers[2] = e1 * powers[1] - 2 * e2
    powers[3] = e1 * powers[2] - e2 * powers[1] + 3 * e3
    for exponent in range(4, spec.power + 1):
        powers[exponent] = (
            e1 * powers[exponent - 1]
            - e2 * powers[exponent - 2]
            + e3 * powers[exponent - 3]
        )
    return powers[spec.power]


def _paths_between(start: tuple[int, int], end: tuple[int, int]) -> int:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx < 0 or dy < 0:
        return 0
    return math.comb(dx + dy, dx)


def _solve_lattice_paths(spec: LatticePathsSpec) -> int:
    origin = (0, 0)
    end = (spec.width, spec.height)
    first = spec.first_forbidden
    second = spec.second_forbidden
    total = _paths_between(origin, end)
    through_first = _paths_between(origin, first) * _paths_between(first, end)
    through_second = _paths_between(origin, second) * _paths_between(second, end)
    through_both = (
        _paths_between(origin, first)
        * _paths_between(first, second)
        * _paths_between(second, end)
    )
    return (total - through_first - through_second + through_both) % 1000


def _solve_modular_power(spec: ModularPowerSpec) -> int:
    exponent = spec.exponent_base**spec.exponent_power + spec.exponent_offset
    return pow(spec.base, exponent, 1000)


def _solve_recurrence(spec: RecurrenceSpec) -> int:
    value = spec.first % spec.modulus
    for _ in range(1, spec.index):
        value = (spec.multiplier * value + spec.increment) % spec.modulus
    return value


def _solve_recurrence_affine(spec: RecurrenceSpec) -> int:
    def compose(
        outer: tuple[int, int], inner: tuple[int, int]
    ) -> tuple[int, int]:
        return (
            (outer[0] * inner[0]) % spec.modulus,
            (outer[0] * inner[1] + outer[1]) % spec.modulus,
        )

    result = (1, 0)
    power = (spec.multiplier % spec.modulus, spec.increment % spec.modulus)
    remaining = spec.index - 1
    while remaining:
        if remaining & 1:
            result = compose(power, result)
        power = compose(power, power)
        remaining >>= 1
    return (result[0] * spec.first + result[1]) % spec.modulus


def _solve_exactly_one(spec: ExactlyOneDivisorSpec) -> int:
    a, b, c = spec.divisors
    singles = sum(spec.limit // divisor for divisor in (a, b, c))
    pairs = sum(
        spec.limit // math.lcm(left, right)
        for left, right in ((a, b), (a, c), (b, c))
    )
    triple = spec.limit // math.lcm(a, b, c)
    return singles - 2 * pairs + 3 * triple


def _solve(spec: OlympiadSpec) -> int:
    if isinstance(spec, ParallelChordsSpec):
        return _solve_parallel_chords(spec)
    if isinstance(spec, CevianSpec):
        return _solve_cevian(spec)
    if isinstance(spec, SymmetricRootsSpec):
        return _solve_symmetric_roots(spec)
    if isinstance(spec, LatticePathsSpec):
        return _solve_lattice_paths(spec)
    if isinstance(spec, ModularPowerSpec):
        return _solve_modular_power(spec)
    if isinstance(spec, RecurrenceSpec):
        return _solve_recurrence(spec)
    if isinstance(spec, ExactlyOneDivisorSpec):
        return _solve_exactly_one(spec)
    raise TypeError(f"unsupported olympiad spec: {type(spec).__name__}")


def _render_parallel_chords(spec: ParallelChordsSpec) -> str:
    if spec.surface % 2 == 0:
        return (
            f"A circle has radius {spec.radius}. Two parallel chords of lengths "
            f"{spec.first_chord} and {spec.second_chord} lie on opposite sides of the "
            "center. Find the distance between the two chords."
        )
    return (
        f"Inside a circular window of radius {spec.radius}, two parallel support bars "
        f"form chords of lengths {spec.first_chord} and {spec.second_chord}. The bars "
        "are on opposite sides of the center. How far apart are the bars?"
    )


def _render_cevian(spec: CevianSpec) -> str:
    dc = spec.side_bc - spec.segment_bd
    if spec.surface % 2 == 0:
        return (
            f"In triangle ABC, point D lies on side BC. The side lengths are "
            f"AB={spec.side_ab}, AC={spec.side_ac}, and BC={spec.side_bc}, while "
            f"BD={spec.segment_bd}. Determine the length AD."
        )
    return (
        f"A triangular plot ABC has AB={spec.side_ab}, AC={spec.side_ac}, and "
        f"BC={spec.side_bc}. A marker D divides BC into lengths BD={spec.segment_bd} "
        f"and DC={dc}. Find the length of the segment from A to D."
    )


def _render_symmetric_roots(spec: SymmetricRootsSpec) -> str:
    polynomial = (
        f"t^3 - {spec.sum_roots}t^2 + {spec.sum_pairwise}t - {spec.product}"
    )
    if spec.surface % 2 == 0:
        return (
            f"The polynomial {polynomial} has positive integer roots x, y, and z. "
            f"Compute x^{spec.power} + y^{spec.power} + z^{spec.power}."
        )
    return (
        "Three positive integers x, y, and z satisfy "
        f"x+y+z={spec.sum_roots}, xy+yz+zx={spec.sum_pairwise}, and "
        f"xyz={spec.product}. Find x^{spec.power}+y^{spec.power}+z^{spec.power}."
    )


def _render_lattice_paths(spec: LatticePathsSpec) -> str:
    p, q = spec.first_forbidden, spec.second_forbidden
    if spec.surface % 2 == 0:
        return (
            f"A robot moves from (0,0) to ({spec.width},{spec.height}), taking only "
            "unit steps right or up. It may not visit either "
            f"({p[0]},{p[1]}) or ({q[0]},{q[1]}). Find the number of valid paths "
            "modulo 1000."
        )
    return (
        f"On a {spec.width}-by-{spec.height} street grid, a courier travels from the "
        "southwest corner to the northeast corner using only north and east moves. "
        f"The intersections ({p[0]},{p[1]}) and ({q[0]},{q[1]}) are closed. How many "
        "routes remain? Give the remainder when the count is divided by 1000."
    )


def _render_modular_power(spec: ModularPowerSpec) -> str:
    exponent = f"{spec.exponent_base}^{spec.exponent_power}+{spec.exponent_offset}"
    if spec.surface % 2 == 0:
        return (
            f"Determine the last three digits of {spec.base}^({exponent}). "
            "Leading zeroes may be included in the final three-digit block."
        )
    return (
        f"An integer N is defined by N={spec.base}^({exponent}). Find the remainder "
        "when N is divided by 1000."
    )


def _render_recurrence(spec: RecurrenceSpec) -> str:
    if spec.surface % 2 == 0:
        return (
            f"A sequence begins with a_1={spec.first} and satisfies "
            f"a_(k+1)={spec.multiplier}a_k+{spec.increment}. Find the remainder when "
            f"a_{spec.index} is divided by {spec.modulus}."
        )
    return (
        f"Starting from {spec.first}, repeatedly multiply the current value by "
        f"{spec.multiplier} and then add {spec.increment}. After {spec.index - 1} "
        f"iterations, what is the resulting value modulo {spec.modulus}?"
    )


def _render_exactly_one(spec: ExactlyOneDivisorSpec) -> str:
    a, b, c = spec.divisors
    if spec.surface % 2 == 0:
        return (
            f"How many positive integers not exceeding {spec.limit} are divisible by "
            f"exactly one of {a}, {b}, and {c}?"
        )
    return (
        f"Tickets are numbered from 1 through {spec.limit}. A ticket is selected if "
        f"its number is a multiple of exactly one of {a}, {b}, or {c}. How many "
        "tickets are selected?"
    )


def _render(spec: OlympiadSpec) -> str:
    if isinstance(spec, ParallelChordsSpec):
        body = _render_parallel_chords(spec)
    elif isinstance(spec, CevianSpec):
        body = _render_cevian(spec)
    elif isinstance(spec, SymmetricRootsSpec):
        body = _render_symmetric_roots(spec)
    elif isinstance(spec, LatticePathsSpec):
        body = _render_lattice_paths(spec)
    elif isinstance(spec, ModularPowerSpec):
        body = _render_modular_power(spec)
    elif isinstance(spec, RecurrenceSpec):
        body = _render_recurrence(spec)
    elif isinstance(spec, ExactlyOneDivisorSpec):
        body = _render_exactly_one(spec)
    else:
        raise TypeError(f"unsupported olympiad spec: {type(spec).__name__}")
    return (
        f"{body}\n\nShow your reasoning clearly, then give the final integer answer as "
        "\\boxed{n}."
    )


def _lattice_paths_dynamic(spec: LatticePathsSpec) -> int:
    forbidden = {spec.first_forbidden, spec.second_forbidden}
    counts: dict[tuple[int, int], int] = {(0, 0): 1}
    for x in range(spec.width + 1):
        for y in range(spec.height + 1):
            point = (x, y)
            if point == (0, 0):
                continue
            if point in forbidden:
                counts[point] = 0
                continue
            counts[point] = counts.get((x - 1, y), 0) + counts.get((x, y - 1), 0)
    return counts[(spec.width, spec.height)] % 1000


def _quality_gate(spec: OlympiadSpec, answer: int, prompt: str) -> bool:
    if not (2 <= answer <= 1_000_000_000):
        return False
    if not (100 <= len(prompt) <= 1_500):
        return False
    if isinstance(spec, ParallelChordsSpec):
        first = _distance_from_center(spec.radius, spec.first_chord)
        second = _distance_from_center(spec.radius, spec.second_chord)
        return first > 0 and second > 0 and first != second and answer == first + second
    if isinstance(spec, CevianSpec):
        length = answer
        bd = spec.segment_bd
        dc = spec.side_bc - bd
        return (
            bd > 0
            and dc > 0
            and spec.side_ab**2 == bd**2 + length**2
            and spec.side_ac**2 == dc**2 + length**2
        )
    if isinstance(spec, SymmetricRootsSpec):
        return answer == sum(root**spec.power for root in spec.hidden_roots)
    if isinstance(spec, LatticePathsSpec):
        return answer == _lattice_paths_dynamic(spec)
    if isinstance(spec, ModularPowerSpec):
        reduced_exponent = (
            pow(spec.exponent_base, spec.exponent_power, 400) + spec.exponent_offset
        ) % 400
        independent = pow(spec.base, reduced_exponent, 1000)
        return (
            math.gcd(spec.base, 1000) == 1
            and answer == _solve_modular_power(spec)
            and answer == independent
        )
    if isinstance(spec, RecurrenceSpec):
        return (
            0 <= answer < spec.modulus
            and answer == _solve_recurrence_affine(spec)
        )
    if isinstance(spec, ExactlyOneDivisorSpec):
        brute = sum(
            sum(value % divisor == 0 for divisor in spec.divisors) == 1
            for value in range(1, spec.limit + 1)
        )
        return answer == brute
    return False


def _extract_boxed_integer(output: str) -> int | None:
    matches = _BOXED_RE.findall(output)
    if not matches:
        return None
    candidate = matches[-1].strip()
    if _INTEGER_RE.fullmatch(candidate) is None:
        return None
    return int(candidate)


class OlympiadTrack:
    track = "olympiad"

    @functools.lru_cache(maxsize=4096)
    def _instance(self, seed: str) -> OlympiadInstance:
        family_index = _int_seed(seed, "olympiad-family") % len(_BUILDERS)
        family, builder, reasoning_steps = _BUILDERS[family_index]
        for attempt in range(100):
            rng = _rng(seed, f"olympiad:{family}:{attempt}")
            spec = builder(rng, _int_seed(seed, f"surface:{attempt}") % 4)
            try:
                answer = _solve(spec)
                prompt = _render(spec)
                if _quality_gate(spec, answer, prompt):
                    return OlympiadInstance(
                        family=family,
                        spec=spec,
                        prompt=prompt,
                        answer=answer,
                        reasoning_steps=reasoning_steps,
                    )
            except (ArithmeticError, ValueError):
                continue
        raise RuntimeError(f"could not generate a valid {family} problem")

    def render(self, seed: str) -> str:
        return self._instance(seed).prompt

    def verify(self, seed: str, output: str) -> bool:
        candidate = _extract_boxed_integer(output)
        return candidate is not None and candidate == self._instance(seed).answer

    def difficulty(self, seed: str) -> Difficulty:
        instance = self._instance(seed)
        return Difficulty(
            track=self.track,
            params={
                "source": "thinker",
                "generator": instance.family,
                "difficulty_tier": "hard",
                "reasoning_steps": instance.reasoning_steps,
                "verification": "exact",
            },
            target_pass_rate=(0.05, 0.35),
        )

    def min_tokens(self, seed: str) -> int:
        return 128 + 24 * self._instance(seed).reasoning_steps


_track = OlympiadTrack()
register_track(_track)


__all__ = ["OlympiadInstance", "OlympiadTrack"]
