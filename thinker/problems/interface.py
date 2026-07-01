from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


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
