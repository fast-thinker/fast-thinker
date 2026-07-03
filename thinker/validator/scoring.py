from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutResult:
    track: str
    seed: str
    band: str
    score: float
    sample_weight: float = 1.0
    original_verified: bool | None = None
    miner_verified: bool | None = None
    original_completion_len: int | None = None
    miner_completion_len: int | None = None


@dataclass(frozen=True)
class StratifiedScore:
    overall: float
    per_band: dict[str, float]
    coverage_ok: bool
    reason: str | None


def difficulty_band(percentile: float | None, n_bands: int = 4) -> str:
    if percentile is None:
        return "uncalibrated"
    idx = min(n_bands - 1, max(0, int(percentile * n_bands)))
    return f"band-{idx}"


def stratified_score(
    results: list[RolloutResult],
    min_coverage_per_band: int = 1,
    required_bands: set[str] | None = None,
) -> StratifiedScore:
    by_band: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in results:
        by_band[r.band].append((r.score, max(0.0, float(r.sample_weight))))

    if required_bands is not None:
        missing = required_bands - set(by_band)
        if missing:
            return StratifiedScore(
                overall=0.0,
                per_band={},
                coverage_ok=False,
                reason=f"missing required bands: {sorted(missing)}",
            )

    under_covered = sorted(b for b, vals in by_band.items() if len(vals) < min_coverage_per_band)
    if under_covered:
        return StratifiedScore(
            overall=0.0,
            per_band={},
            coverage_ok=False,
            reason=f"under-covered bands: {under_covered}",
        )

    per_band: dict[str, float] = {}
    for band, vals in by_band.items():
        total_weight = sum(weight for _score, weight in vals)
        if total_weight > 0:
            per_band[band] = (
                sum(score * weight for score, weight in vals) / total_weight
            )
        else:
            per_band[band] = sum(score for score, _weight in vals) / len(vals)
    overall = sum(per_band.values()) / len(per_band) if per_band else 0.0
    return StratifiedScore(overall=overall, per_band=per_band, coverage_ok=True, reason=None)
