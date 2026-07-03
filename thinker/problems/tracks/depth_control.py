from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from thinker.problems.interface import Difficulty, register_track

MIN_OP = 8
MAX_OP = 60

_VAR_PATTERN = re.compile(r"V\d{6}")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}", re.DOTALL)


@dataclass(frozen=True)
class _DepthTask:
    op: int
    noise: int
    output_list: list[str]
    query_value: int
    query_list: list[str]
    solution: str


def _int_seed(seed: str, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(seed: str) -> random.Random:
    return random.Random(_int_seed(seed, "depth-control"))


def _symbolic_prompt(query_value: int, context: str) -> str:
    return (
        f"<context>\n{context}\n</context>\n\n"
        "The context contains relationships between variables. These relationships "
        "are independent mathematical equations that are all satisfied simultaneously.\n"
        f"Using only these relationships, determine which variables have values equal to "
        f"{query_value}.\n"
        "Show your step-by-step reasoning and calculations, and then conclude your final "
        "answer in a sentence."
    )


def _var_names(rng: random.Random, n: int) -> list[str]:
    return [f"V{i:06d}" for i in rng.sample(range(1_000_000), k=n)]


def _random_non_target(rng: random.Random, target: int) -> int:
    value = target
    while value == target:
        value = rng.randint(-1_000_000, 1_000_000)
    return value


def _boxed_or_all_vars(output: str) -> set[str]:
    boxed = _BOXED_RE.findall(output)
    text = boxed[-1] if boxed else output
    return set(_VAR_PATTERN.findall(text))


class DepthControlTrack:
    track = "depth_control"

    def _build_task(self, seed: str) -> _DepthTask:
        rng = _rng(seed)
        op = MIN_OP + _int_seed(seed, "op") % (MAX_OP - MIN_OP + 1)
        noise = _int_seed(seed, "noise") % (op + 1)
        variables = _var_names(rng, op + noise)

        values: dict[str, int] = {}
        equations: list[str] = []
        trace: list[str] = []

        current = rng.randint(-1_000, 1_000)
        values[variables[0]] = current
        equations.append(f"{variables[0]} = {current}")
        trace.append(f"{variables[0]} = {current}")

        for i in range(1, op):
            prev = variables[i - 1]
            name = variables[i]
            kind = rng.choice(("add", "sub", "mul"))
            if kind == "add":
                delta = rng.choice([x for x in range(-250, 251) if x != 0])
                current = values[prev] + delta
                equations.append(f"{name} = {prev} + {delta}")
                trace.append(f"{name} = {values[prev]} + {delta} = {current}")
            elif kind == "sub":
                delta = rng.randint(1, 250)
                current = values[prev] - delta
                equations.append(f"{name} = {prev} - {delta}")
                trace.append(f"{name} = {values[prev]} - {delta} = {current}")
            else:
                factor = rng.choice((-9, -7, -5, -3, -2, 2, 3, 5, 7, 9))
                current = values[prev] * factor
                equations.append(f"{name} = {prev} * {factor}")
                trace.append(f"{name} = {values[prev]} * {factor} = {current}")
            values[name] = current

        target_var = variables[_int_seed(seed, "target") % op]
        query_value = values[target_var]

        for i in range(op, op + noise):
            name = variables[i]
            if i == op and rng.random() < 0.4:
                value = query_value
            else:
                value = _random_non_target(rng, query_value)
            values[name] = value
            equations.append(f"{name} = {value}")
            trace.append(f"{name} = {value}")

        rng.shuffle(equations)
        query_list = [name for name in variables if values[name] == query_value]
        solution = "\n".join(trace + [f"ANSWER: {', '.join(query_list)}"])
        return _DepthTask(op, noise, equations, query_value, query_list, solution)

    def _task(self, seed: str):
        task = self._build_task(seed)
        return (
            task.op,
            task.noise,
            task.output_list,
            task.query_value,
            task.query_list,
            task.solution,
        )

    def render(self, seed: str) -> str:
        task = self._build_task(seed)
        context = ". ".join(task.output_list) + "."
        return _symbolic_prompt(task.query_value, context)

    def verify(self, seed: str, output: str) -> bool:
        task = self._build_task(seed)
        return _boxed_or_all_vars(output) == set(task.query_list)

    def difficulty(self, seed: str) -> Difficulty:
        task = self._build_task(seed)
        params = {
            "op": task.op,
            "n_total": task.op + task.noise,
            "noise": task.noise,
            "n_answers": len(task.query_list),
        }
        return Difficulty(track=self.track, params=params)

    def min_tokens(self, seed: str) -> int:
        task = self._build_task(seed)
        return max(64, len(task.solution) // 4)


_track = DepthControlTrack()
register_track(_track)
