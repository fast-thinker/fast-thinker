from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


_BOXED_START_RE = re.compile(r"\\boxed\s*\{")


def extract_final_boxed_answer(text: str) -> str | None:
    """Return the sole final boxed payload, including support for nested braces."""
    starts = list(_BOXED_START_RE.finditer(text))
    if len(starts) != 1:
        return None
    start = starts[0]
    depth = 1
    chars: list[str] = []
    for index in range(start.end(), len(text)):
        char = text[index]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                if text[index + 1 :].strip():
                    return None
                answer = "".join(chars).strip()
                return answer or None
            chars.append(char)
        else:
            chars.append(char)
    return None


@dataclass(frozen=True)
class Difficulty:
    track: str
    params: dict[str, Any] = field(default_factory=dict)
    target_pass_rate: tuple[float, float] | None = None
    calibrated_pass_rate: float | None = None
    percentile: float | None = None


@runtime_checkable
class Problem(Protocol):
    track: str

    def render(self, seed: str) -> str:
        ...

    def verify(self, seed: str, output: str) -> bool:
        ...

    def difficulty(self, seed: str) -> Difficulty:
        ...

    def min_tokens(self, seed: str) -> int:
        ...


@dataclass(frozen=True)
class ProblemInstance:
    track: str
    seed: str
    prompt: str
    difficulty: Difficulty
    min_tokens: int


_REGISTRY: dict[str, Problem] = {}


def register_track(problem: Problem) -> None:
    if problem.track in _REGISTRY:
        raise ValueError(f"track {problem.track!r} already registered")
    _REGISTRY[problem.track] = problem


def unregister_track(name: str) -> None:
    _REGISTRY.pop(name, None)


def get_track(name: str) -> Problem:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown track {name!r}; registered tracks: {sorted(_REGISTRY)}"
        ) from None


def registered_tracks() -> list[str]:
    return sorted(_REGISTRY)


def render_instance(track: str, seed: str) -> ProblemInstance:
    problem = get_track(track)
    return ProblemInstance(
        track=track,
        seed=seed,
        prompt=problem.render(seed),
        difficulty=problem.difficulty(seed),
        min_tokens=problem.min_tokens(seed),
    )
