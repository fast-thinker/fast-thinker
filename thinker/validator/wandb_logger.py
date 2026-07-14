from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from thinker.validator.epoch_loop import MinerEpochResult


DASHBOARD_SCHEMA_VERSION = 2


class WandbRun(Protocol):
    def log(self, data: dict[str, Any]) -> None: ...


class WandbEpochLogger:
    def __init__(self, run: WandbRun):
        self._run = run
        self._progress: dict[str, Any] | None = None

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _initialize_progress(self, epoch: int, miner_hotkeys) -> None:
        miners = {str(hotkey): "pending" for hotkey in miner_hotkeys}
        self._progress = {
            "schema_version": 1,
            "epoch": int(epoch),
            "status": "running",
            "current_stage": "qualification",
            "updated_at": self._timestamp(),
            "stages": {
                "qualification": {
                    "status": "pending",
                    "baseline": "pending",
                    "miners": dict(miners),
                    "scores": {},
                },
                "full_evaluation": {
                    "status": "waiting",
                    "baseline": "waiting",
                    "miners": {hotkey: "waiting" for hotkey in miners},
                    "scores": {},
                },
            },
        }

    def _progress_fields(self) -> dict[str, Any]:
        if self._progress is None:
            return {}
        self._progress["updated_at"] = self._timestamp()
        return {
            "progress/epoch": self._progress["epoch"],
            "progress/status": self._progress["status"],
            "progress/updated_at": self._progress["updated_at"],
            "progress/snapshot": json.dumps(
                self._progress, separators=(",", ":"), sort_keys=True
            ),
        }

    def _publish_progress(self) -> None:
        if self._progress is None:
            return
        self._run.log({"epoch": self._progress["epoch"], **self._progress_fields()})

    def start_epoch(self, epoch: int, miner_hotkeys) -> None:
        self._initialize_progress(epoch, miner_hotkeys)
        self._publish_progress()

    def log_progress(self, event: dict[str, Any]) -> None:
        epoch = int(event["epoch"])
        miner_updates = {
            str(hotkey): str(status)
            for hotkey, status in dict(event.get("miner_updates") or {}).items()
        }
        if self._progress is None or self._progress.get("epoch") != epoch:
            self._initialize_progress(epoch, miner_updates)

        stage_name = str(event["stage"])
        stages = self._progress["stages"]
        stage = stages.setdefault(stage_name, {"status": "pending", "miners": {}, "scores": {}})
        if event.get("stage_status") is not None:
            stage["status"] = str(event["stage_status"])
        if event.get("baseline_status") is not None:
            stage["baseline"] = str(event["baseline_status"])
        stage["miners"].update(miner_updates)
        miner_scores = {
            str(hotkey): float(score)
            for hotkey, score in dict(event.get("miner_scores") or {}).items()
        }
        if miner_scores:
            stage.setdefault("scores", {}).update(miner_scores)
        self._progress["status"] = str(event.get("status") or "running")
        self._progress["current_stage"] = stage_name
        self._publish_progress()

    def fail_epoch(self, epoch: int) -> None:
        if self._progress is None or self._progress.get("epoch") != int(epoch):
            self._initialize_progress(epoch, [])
        self._progress["status"] = "failed"
        current_stage = self._progress.get("current_stage")
        if current_stage in self._progress["stages"]:
            stage = self._progress["stages"][current_stage]
            stage["status"] = "failed"
            if stage.get("baseline") in {"pending", "waiting", "preparing", "evaluating"}:
                stage["baseline"] = "failed"
        self._publish_progress()

    def log_epoch(self, epoch: int, results: dict[str, MinerEpochResult]) -> None:
        data: dict[str, Any] = {
            "epoch": epoch,
            "evaluation/round_complete": 1,
            "evaluation/miners_seen": len(results),
        }
        scored_results = []
        component_rows: list[dict[str, Any]] = []
        for miner_hotkey, result in results.items():
            score = None if result.score is None else float(result.score.overall)
            if score is not None and math.isfinite(score):
                data[f"miner/{miner_hotkey}/score"] = score
                if result.completion_len is not None:
                    data[f"miner/{miner_hotkey}/completion_len"] = result.completion_len
                if result.correctness_score is not None:
                    data[f"miner/{miner_hotkey}/correctness_score"] = (
                        result.correctness_score
                    )
                for task, task_score in sorted(result.task_scores.items()):
                    data[f"task/{task}/miner/{miner_hotkey}/score"] = task_score
                for task, length in sorted(result.task_completion_len.items()):
                    data[f"task/{task}/miner/{miner_hotkey}/completion_len"] = length
                for task, correctness in sorted(result.task_correctness_score.items()):
                    data[f"task/{task}/miner/{miner_hotkey}/correctness_score"] = (
                        correctness
                    )
                component_rows.append(
                    {
                        "miner": str(miner_hotkey),
                        "score": score,
                        "math": result.task_scores.get("math"),
                        "long_qa": result.task_scores.get("long_context_qa"),
                        "science": result.task_scores.get("multiple_choice"),
                    }
                )
                scored_results.append(result)

        data["evaluation/miners_scored"] = len(scored_results)
        data["evaluation/miners_rejected"] = len(results) - len(scored_results)
        if component_rows:
            component_rows.sort(key=lambda row: (-float(row["score"]), row["miner"]))
            data["evaluation/component_scores"] = json.dumps(
                component_rows, separators=(",", ":"), sort_keys=True
            )

        # Staged evaluation can use different batches for different miners. Match
        # the shared baseline to the epoch's highest-scoring miner so the dashboard
        # compares its top-miner trend with the baseline for that miner's batch.
        baseline_result = max(
            (
                result
                for result in scored_results
                if result.original_score is not None
                or result.original_correctness_score is not None
                or result.original_completion_len is not None
            ),
            key=lambda result: result.score.overall,
            default=None,
        )
        if baseline_result is not None:
            if baseline_result.original_score is not None:
                data["original/score"] = baseline_result.original_score
            if baseline_result.original_correctness_score is not None:
                data["original/correctness_score"] = (
                    baseline_result.original_correctness_score
                )
            if baseline_result.original_completion_len is not None:
                data["original/completion_len"] = (
                    baseline_result.original_completion_len
                )
            for task, score in sorted(baseline_result.task_original_score.items()):
                data[f"task/{task}/original/score"] = score
            for task, correctness in sorted(
                baseline_result.task_original_correctness_score.items()
            ):
                data[f"task/{task}/original/correctness_score"] = correctness
            for task, length in sorted(
                baseline_result.task_original_completion_len.items()
            ):
                data[f"task/{task}/original/completion_len"] = length
        if self._progress is not None and self._progress.get("epoch") == int(epoch):
            for stage_name, stage in self._progress["stages"].items():
                if stage["status"] in {"pending", "waiting", "preparing", "evaluating"}:
                    stage["status"] = (
                        "skipped"
                        if stage_name == "full_evaluation"
                        and stage["status"] in {"pending", "waiting"}
                        else "completed"
                    )
                for hotkey, miner_status in list(stage["miners"].items()):
                    result = results.get(hotkey)
                    if miner_status == "evaluating":
                        stage["miners"][hotkey] = (
                            "finished"
                            if result is not None and result.score is not None
                            else "failed"
                        )
                    elif miner_status in {"pending", "waiting"}:
                        stage["miners"][hotkey] = (
                            "skipped" if stage_name == "qualification" else "not_selected"
                        )
                if stage.get("baseline") in {"pending", "waiting", "preparing", "evaluating"}:
                    stage["baseline"] = (
                        "skipped"
                        if stage_name == "full_evaluation"
                        and stage["status"] == "skipped"
                        else "finished"
                    )
            self._progress["status"] = "completed"
            self._progress["current_stage"] = None
            data.update(self._progress_fields())
        self._run.log(data)

    def close(self) -> None:
        finish = getattr(self._run, "finish", None)
        if callable(finish):
            finish()


def resolve_wandb_project(
    project: str, entity: str | None = None
) -> tuple[str, str | None]:
    project = project.strip()
    if "/" in project:
        path_entity, project_name = project.rsplit("/", 1)
        if not path_entity or not project_name:
            raise ValueError("W&B project must be 'project' or 'entity/project'")
        return project_name, entity or path_entity
    if not project:
        raise ValueError("W&B project cannot be empty")
    return project, entity or None


def init_validator_run(
    project: str,
    validator_hotkey: str,
    *,
    entity: str | None = None,
) -> WandbRun:
    import wandb

    project, entity = resolve_wandb_project(project, entity)
    run = wandb.init(
        project=project,
        entity=entity or None,
        name=f"validator-{validator_hotkey}",
        group=f"validator-{validator_hotkey}",
        config={
            "validator_hotkey": validator_hotkey,
            "dashboard_schema_version": DASHBOARD_SCHEMA_VERSION,
        },
        mode="online",
        reinit=True,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    return run


__all__ = [
    "DASHBOARD_SCHEMA_VERSION",
    "WandbEpochLogger",
    "WandbRun",
    "init_validator_run",
    "resolve_wandb_project",
]
