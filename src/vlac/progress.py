"""Console progress, approximate timing and durable run status."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid


def _json_default(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, complex):
            return {"real": scalar.real, "imag": scalar.imag}
        return scalar
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "estimating"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Report stage progress to stderr and retain JSONL/status checkpoints.

    Stage ETAs are computed from elapsed time per completed unit. A full-run
    ETA is shown only when prior timing data covers all planned future stages;
    otherwise the reporter says that the total is still being calibrated.
    """

    def __init__(self, command: str, output_dir: str | Path, *, context: dict | None = None):
        self.command = command
        self.context = context or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "progress.jsonl"
        self.status_path = self.output_dir / "run_status.json"
        self.history_path = self.output_dir / "runtime_history.json"
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        self.started = time.monotonic()
        self.stage_name: str | None = None
        self.stage_started: float | None = None
        self.stage_total: int | None = None
        self.stage_unit = "items"
        self.stage_current = 0
        self._last_console = 0.0
        self._last_percent = -1
        self._solver_started = 0
        self._solver_completed = 0
        self._solver_failed = 0
        self._solver_total: int | None = None
        self._solver_durations: deque[float] = deque(maxlen=12)
        self._timings = {}
        self._cache_stats: dict[str, dict[str, int]] = {}
        self._stage_solver_start = (0, 0, 0)
        self._stage_cache_start: dict[str, dict[str, int]] = {}
        self._completed_stages: list[str] = []
        self._checkpoints: dict[str, object] = {}
        self._planned_stages: list[str] = []
        self._stage_durations: dict[str, float] = {}
        self._history = self._load_history()
        self._write_status("running")
        self._record("run_started", {"command": command})
        self._console(f"Started {command}; progress log: {self.log_path}")

    def _load_history(self) -> dict:
        try:
            raw = json.loads(self.history_path.read_text(encoding="utf-8"))
            if raw.get("schema_version") == 1:
                return raw.get("commands", {})
        except (OSError, ValueError, AttributeError):
            pass
        return {}

    def set_plan(self, stage_names: list[str]) -> None:
        self._planned_stages = list(stage_names)

    @property
    def completed_stages(self) -> list[str]:
        return list(self._completed_stages)

    def set_solver_total(self, total: int | None) -> None:
        self._solver_total = total
        self._write_status("running")

    def cache_event(self, kind: str, event: str) -> None:
        """Accumulate cache events without logging every potentially huge lookup."""
        counters = self._cache_stats.setdefault(
            kind, {"lookups": 0, "hits": 0, "misses": 0, "writes": 0}
        )
        if event == "write":
            counters["writes"] += 1
            return
        counters["lookups"] += 1
        counters["hits" if event == "hit" else "misses"] += 1

    def record_timing(self, category: str, seconds: float) -> None:
        row = self._timings.setdefault(category, {"calls": 0, "seconds": 0.0})
        row["calls"] += 1
        row["seconds"] += seconds

    @property
    def runtime_stats(self) -> dict:
        return {
            "timings": {name: {"calls": row["calls"], "seconds": round(row["seconds"], 6)}
                        for name, row in self._timings.items()},
            "findmodes_calls": {
                "started": self._solver_started,
                "completed": self._solver_completed,
                "failed": self._solver_failed,
            },
            "cache": {
                kind: dict(values) for kind, values in self._cache_stats.items()
            },
        }

    def _stage_diagnostics(self) -> dict:
        start_started, start_completed, start_failed = self._stage_solver_start
        cache_delta = {}
        for kind, current in self._cache_stats.items():
            start = self._stage_cache_start.get(kind, {})
            cache_delta[kind] = {
                name: current[name] - start.get(name, 0)
                for name in ("lookups", "hits", "misses", "writes")
            }
        return {
            "findmodes_calls": {
                "started": self._solver_started - start_started,
                "completed": self._solver_completed - start_completed,
                "failed": self._solver_failed - start_failed,
            },
            "cache": cache_delta,
        }

    def _record(self, event: str, fields: dict | None = None) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "command": self.command,
            "event": event,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "stage": self.stage_name,
            "context": self.context,
        }
        if fields:
            record.update(fields)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=_json_default) + "\n")

    def _write_status(self, status: str, error: str | None = None) -> None:
        data = {
            "schema_version": 1,
            "run_id": self.run_id,
            "command": self.command,
            "context": self.context,
            "status": status,
            "started_utc": datetime.fromtimestamp(
                time.time() - (time.monotonic() - self.started), timezone.utc
            ).isoformat(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "current_stage": self.stage_name,
            "stage_current": self.stage_current,
            "stage_total": self.stage_total,
            "stage_unit": self.stage_unit,
            "completed_stages": self._completed_stages,
            "fde_solves_completed": self._solver_completed,
            "fde_solves_started": self._solver_started,
            "fde_solves_expected": self._solver_total,
            "findmodes_calls": self.runtime_stats["findmodes_calls"],
            "cache_stats": self.runtime_stats["cache"],
            "timings": self.runtime_stats["timings"],
            "current_stage_diagnostics": self._stage_diagnostics(),
            "checkpoints": self._checkpoints,
            "progress_log": str(self.log_path),
        }
        if error is not None:
            data["error"] = error
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(data, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.status_path)

    def _console(self, message: str) -> None:
        elapsed = _duration(time.monotonic() - self.started)
        print(f"[{elapsed} elapsed] {message}", file=sys.stderr, flush=True)
        self._last_console = time.monotonic()

    def _previous_stage_estimate(self, stage_name: str) -> float | None:
        samples = self._history.get(self.command, {}).get(stage_name, [])
        valid = [float(value) for value in samples if float(value) > 0]
        return sum(valid[-5:]) / min(len(valid), 5) if valid else None

    def _overall_eta(self, current_stage_eta: float | None) -> tuple[float | None, bool]:
        if current_stage_eta is None or not self._planned_stages:
            return None, False
        try:
            index = self._planned_stages.index(self.stage_name)
        except ValueError:
            return None, False
        total = current_stage_eta
        for name in self._planned_stages[index + 1 :]:
            estimate = self._previous_stage_estimate(name)
            if estimate is None:
                return total, False
            total += estimate
        return total, True

    def start_stage(self, name: str, total: int | None = None, unit: str = "items") -> None:
        if self.stage_name is not None:
            self.finish_stage("stage advanced")
        self.stage_name = name
        self.stage_started = time.monotonic()
        self.stage_total = total
        self.stage_unit = unit
        self.stage_current = 0
        self._stage_solver_start = (
            self._solver_started,
            self._solver_completed,
            self._solver_failed,
        )
        self._stage_cache_start = {
            kind: dict(values) for kind, values in self._cache_stats.items()
        }
        self._last_percent = -1
        self._record("stage_started", {"total": total, "unit": unit})
        suffix = f" ({total} {unit})" if total is not None else ""
        self._console(f"{name}: started{suffix}")
        self._write_status("running")

    def update(self, current: int, total: int | None = None, detail: str | None = None) -> None:
        if total is not None:
            self.stage_total = total
        self.stage_current = int(current)
        eta_stage = None
        percent = None
        if self.stage_total:
            percent = min(100, int(100 * self.stage_current / self.stage_total))
            if self.stage_current >= 2 and self.stage_current < self.stage_total and self.stage_started:
                elapsed = time.monotonic() - self.stage_started
                eta_stage = elapsed / self.stage_current * (self.stage_total - self.stage_current)
        should_print = (
            self.stage_current <= 1
            or (percent is not None and percent >= self._last_percent + 5)
            or (self.stage_total is not None and self.stage_current >= self.stage_total)
            or time.monotonic() - self._last_console >= 10
        )
        fields = {
            "current": self.stage_current,
            "total": self.stage_total,
            "unit": self.stage_unit,
            "eta_stage_seconds": None if eta_stage is None else round(eta_stage, 1),
            "detail": detail,
        }
        self._record("progress", fields)
        if should_print:
            progress_text = (
                f"{self.stage_current}/{self.stage_total} {self.stage_unit}"
                if self.stage_total is not None
                else f"{self.stage_current} {self.stage_unit}"
            )
            suffix = f"; stage ETA ~{_duration(eta_stage)}" if eta_stage is not None else ""
            if self._solver_total is not None and self._solver_completed >= 2 and self._solver_durations:
                average = sum(self._solver_durations) / len(self._solver_durations)
                eta_solver = average * max(0, self._solver_total - self._solver_completed)
                suffix += f"; total FDE ETA ~{_duration(eta_solver)}"
            stage_diagnostics = self._stage_diagnostics()
            calls = stage_diagnostics["findmodes_calls"]
            if calls["started"]:
                suffix += (
                    f"; findmodes {calls['completed']} completed"
                    f"/{calls['started']} started this stage"
                )
            lookups = sum(v["lookups"] for v in stage_diagnostics["cache"].values())
            hits = sum(v["hits"] for v in stage_diagnostics["cache"].values())
            misses = sum(v["misses"] for v in stage_diagnostics["cache"].values())
            if lookups:
                suffix += f"; cache {hits}/{lookups} hits ({misses} misses) this stage"
            total_eta, complete = self._overall_eta(eta_stage)
            if total_eta is not None:
                suffix += (
                    f"; overall ETA ~{_duration(total_eta)}"
                    if complete else f"; overall ETA at least ~{_duration(total_eta)}"
                )
            if detail:
                suffix += f"; {detail}"
            self._console(f"{self.stage_name}: {progress_text}{suffix}")
            if percent is not None:
                self._last_percent = percent
        self._write_status("running")

    def note(self, message: str, *, force: bool = False) -> None:
        self._record("message", {"message": message})
        if force or time.monotonic() - self._last_console >= 10:
            self._console(f"{self.stage_name or self.command}: {message}")
        self._write_status("running")

    def native_solves(self, count):
        """Record solver-reported native attempts; avoid fabricated per-solve timings."""
        self._solver_started += count
        self._solver_completed += count
        self._record('native_solve_batch', {'attempts': count, 'completed': self._solver_completed})
        self._write_status('running')

    def solver_started(self) -> float:
        self._solver_started += 1
        token = time.monotonic()
        should_print = (
            self._solver_started <= 2
            or self._solver_started % 10 == 0
            or time.monotonic() - self._last_console >= 10
        )
        fields = {"solve": self._solver_started, "expected": self._solver_total}
        self._record("fde_solve_started", fields)
        if should_print:
            count = f"{self._solver_started}/{self._solver_total}" if self._solver_total else str(self._solver_started)
            self._console(f"{self.stage_name or self.command}: MODE findmodes call {count} started")
        self._write_status("running")
        return token

    def solver_finished(self, token: float, error: str | None = None) -> None:
        duration = max(0.0, time.monotonic() - token)
        self._solver_durations.append(duration)
        if error is None:
            self._solver_completed += 1
        else:
            self._solver_failed += 1
        self._record(
            "fde_solve_failed" if error else "fde_solve_completed",
            {
                "solve": self._solver_started,
                "duration_seconds": round(duration, 3),
                "error": error,
            },
        )
        average = sum(self._solver_durations) / len(self._solver_durations)
        eta = (
            average * max(0, self._solver_total - self._solver_completed)
            if self._solver_total is not None else None
        )
        should_print = (
            self._solver_completed <= 2
            or self._solver_completed % 10 == 0
            or (self._solver_total is not None and self._solver_completed >= self._solver_total)
            or time.monotonic() - self._last_console >= 10
        )
        if should_print:
            count = f"{self._solver_completed}/{self._solver_total}" if self._solver_total else str(self._solver_completed)
            suffix = f"; total FDE ETA ~{_duration(eta)}" if eta is not None else f"; mean solve {_duration(average)}"
            self._console(f"{self.stage_name or self.command}: MODE findmodes call {count} complete{suffix}")
        self._write_status("running", error=error if error else None)

    def checkpoint(self, name: str, payload: object) -> None:
        self._checkpoints[name] = payload
        self._record("checkpoint", {"name": name, "payload": payload})
        self._write_status("running")

    def finish_stage(self, detail: str | None = None) -> None:
        if self.stage_name is None:
            return
        duration = max(0.0, time.monotonic() - (self.stage_started or time.monotonic()))
        name = self.stage_name
        self._stage_durations[name] = duration
        self._completed_stages.append(name)
        diagnostics = self._stage_diagnostics()
        self._record(
            "stage_completed",
            {
                "duration_seconds": round(duration, 3),
                "detail": detail,
                "runtime_diagnostics": diagnostics,
            },
        )
        calls = diagnostics["findmodes_calls"]
        cache_lookups = sum(v["lookups"] for v in diagnostics["cache"].values())
        cache_hits = sum(v["hits"] for v in diagnostics["cache"].values())
        suffix = (
            f"; {calls['completed']} findmodes calls"
            f" ({calls['failed']} failed)"
        )
        if cache_lookups:
            suffix += f"; {cache_hits}/{cache_lookups} cache hits"
        if detail:
            suffix += f"; {detail}"
        self._console(f"{name}: complete in {_duration(duration)}{suffix}")
        self.stage_name = None
        self.stage_started = None
        self.stage_total = None
        self.stage_current = 0
        self._write_status("running")

    def _save_history(self) -> None:
        commands = dict(self._history)
        command_data = dict(commands.get(self.command, {}))
        for name, duration in self._stage_durations.items():
            samples = list(command_data.get(name, []))
            samples.append(round(duration, 3))
            command_data[name] = samples[-5:]
        commands[self.command] = command_data
        temporary = self.history_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"schema_version": 1, "commands": commands}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.history_path)

    def finish(self) -> None:
        self.finish_stage()
        self._save_history()
        self._record(
            "run_completed",
            {
                "completed_stages": self._completed_stages,
                "runtime_diagnostics": self.runtime_stats,
            },
        )
        self._write_status("completed")
        self._console(f"{self.command}: completed; status: {self.status_path}")

    def fail(self, exc: BaseException) -> None:
        self._record(
            "run_failed",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime_diagnostics": self.runtime_stats,
            },
        )
        self._write_status("failed", error=f"{type(exc).__name__}: {exc}")
        self._console(f"{self.command}: failed in {self.stage_name or 'startup'}: {type(exc).__name__}: {exc}")
