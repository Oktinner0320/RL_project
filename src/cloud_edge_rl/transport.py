"""Bounded, restartable artifact transport; never submits compute or hashes files."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class TransportLimits:
    listing_seconds: int = 45
    file_seconds: int = 45
    download_seconds: int = 180
    cleanup_seconds: int = 90
    verification_seconds: int = 60
    status_reads: int = 6
    status_interval_seconds: int = 2
    read_attempts: int = 2
    max_entries: int = 32
    max_file_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("Transport limits must be positive integers")
        if self.read_attempts > 3 or self.max_file_bytes > self.max_total_bytes:
            raise ValueError("Unsafe transport limits")


ARTIFACT_FILES = frozenset({
    "config.json", "metadata.json", "runtime.json", "data_metadata.json", "status.json",
    "summary.json", "policy_training.json", "training.pt", "agent.pt", "model.pt", "risk.pt",
    "metrics.jsonl", "metrics.csv", "reference_sources.json", "error.txt",
    "policy_decisions.jsonl", "uploads.jsonl", "audits.jsonl", "risk_dataset.csv",
    "risk_metrics.json", "calibration.json", "audit_summary.json", "resolved_audit_config.json",
    "risk_predictions.csv", "calibration_timeline.csv", "calibration_metadata.json",
    "environment.pt", "checkpoint.pt",
})
REMOTE_ONLY_DIRECTORIES = frozenset({
    "uploads", "reference_uploads", "referenceuploads", "private", "updates", "references",
    "attack_details",
})


def read_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=invalid, parse_float=finite_float)


def write_json(path: Path, value, *, replace: bool = False):
    """Publish a complete fsynced document, never a half-written success marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Refuse symbolic artifact path")
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(partial, path)
    else:
        os.link(partial, path)
        partial.unlink()


def safe_run_id(run_id: str) -> str:
    if (type(run_id) is not str or not run_id or run_id in (".", "..")
            or PurePosixPath(run_id).name != run_id or "\\" in run_id or ":" in run_id):
        raise ValueError("Safe exact run ID required")
    return run_id


def run_bounded(command: list[str], timeout: float, log_path: Path, *, stdout_path=None) -> dict:
    """Watch descendants even after their parent exits; kill/reap on every exit path."""
    import psutil

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Positive finite process timeout required")
    started = time.monotonic()
    children = {}
    timed_out = False
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
    }
    from contextlib import ExitStack
    with ExitStack() as stack:
        log = stack.enter_context(log_path.open("xb"))
        output = log if stdout_path is None else stack.enter_context(Path(stdout_path).open("xb"))
        process = subprocess.Popen(command, stdout=output,
                                   stderr=subprocess.STDOUT if output is log else log, **options)
        parent = psutil.Process(process.pid)
        try:
            while True:
                try:
                    for child in parent.children(recursive=True):
                        children[child.pid] = child
                except psutil.NoSuchProcess:
                    pass
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(0.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            try:
                for child in parent.children(recursive=True):
                    children[child.pid] = child
            except psutil.NoSuchProcess:
                pass
            targets = list(children.values())
            if process.poll() is None:
                targets.append(parent)
            for target in reversed(targets):
                try:
                    target.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(targets, timeout=5)
            process.wait(timeout=5)
    record = {"pid": process.pid, "exit_code": process.returncode, "timed_out": timed_out,
              "elapsed_seconds": time.monotonic() - started,
              "process_tree_clean": not alive, "surviving_pids": [p.pid for p in alive],
              "log_path": str(log_path)}
    write_json(log_path.with_suffix(".json"), record)
    if alive:
        raise RuntimeError(f"Transport process tree not clean: {record['surviving_pids']}")
    if timed_out:
        raise TimeoutError(f"Transport worker deadline exceeded: {log_path}")
    if process.returncode:
        raise RuntimeError(f"Transport worker exit {process.returncode}: {log_path}")
    return record


def stop_and_verify(app_id: str, *, environment: str = "main", limits=None,
                    command=None, clock=time.monotonic, sleep=time.sleep) -> dict:
    """Stop the exact app, then wait for an observed stopped state, not a guessed delay."""
    limits = limits or TransportLimits()
    if type(app_id) is not str or not app_id.startswith("ap-"):
        raise ValueError("Exact Modal app ID required")
    command = command or _cli
    start = clock()
    stop_error = None
    try:
        command(["app", "stop", "-y", "--env", environment, app_id],
                min(20, limits.cleanup_seconds))
    except Exception as error:
        # A failed stop command is not proof of state (the app may already be stopped).
        stop_error = f"{type(error).__name__}: {error}"
    last = None
    for attempt in range(limits.status_reads):
        remaining = limits.cleanup_seconds - (clock() - start)
        if remaining <= 0:
            break
        rows = json.loads(command(["app", "list", "--env", environment, "--json",
                                   "--limit", "100"], min(15, remaining)))
        if (type(rows) is not list or any(type(row) is not dict
                or type(row.get("app_id")) is not str or type(row.get("state")) is not str
                for row in rows)):
            raise ValueError("Unknown app status schema")
        matching = [row for row in rows if row["app_id"] == app_id]
        if len(matching) != 1:
            raise ValueError("Exact app absent or duplicated; shutdown remains unknown")
        last = matching[0]["state"].lower()
        if last == "stopped" and clock() - start < limits.cleanup_seconds:
            receipt = {"app_id": app_id, "verified": True, "stopped": True,
                       "status_reads": attempt + 1, "elapsed_seconds": clock() - start}
            if stop_error is not None:
                receipt["stop_error"] = stop_error
            return receipt
        if last not in {"stopping", "running", "active", "ephemeral", "detached",
                        "initializing", "deployed"}:
            raise ValueError(f"Unknown app state: {last}")
        remaining = limits.cleanup_seconds - (clock() - start)
        if attempt + 1 < limits.status_reads and remaining > 0:
            sleep(min(limits.status_interval_seconds, remaining))
    raise TimeoutError(f"App shutdown not verified; last observed state={last}")


def _cli(arguments, timeout):
    import tempfile
    approved = Path.home() / "AppData" / "Local" / "Temp" / "opencode"
    directory = Path(tempfile.mkdtemp(prefix="rl-modal-status-",
                                     dir=approved if approved.is_dir() else None))
    output = directory / "stdout.txt"
    run_bounded([sys.executable, "-X", "utf8", "-m", "modal", *arguments], timeout,
                directory / "command.log", stdout_path=output)
    if output.stat().st_size > 1024 * 1024:
        raise ValueError("Modal status output exceeds bound")
    return output.read_text(encoding="utf-8")


def inventory_entries(entries, run_id: str, limits: TransportLimits) -> dict:
    prefix = "runs/" + safe_run_id(run_id)
    selected, skipped, seen = [], [], set()
    total = 0
    for entry in entries:
        path = entry.path.lstrip("/")
        relative = PurePosixPath(path).relative_to(PurePosixPath(prefix))
        if (len(seen) >= limits.max_entries or len(relative.parts) != 1 or "\\" in path
                or relative.name in (".", "..") or path != prefix + "/" + relative.name
                or path in seen):
            raise ValueError("Unsafe, duplicate or excessive top-level artifact listing")
        seen.add(path)
        kind = int(entry.type)
        if kind == 2 and relative.name in REMOTE_ONLY_DIRECTORIES:
            skipped.append(path)
            continue
        if kind != 1 or relative.name not in ARTIFACT_FILES:
            raise ValueError(f"Unexpected remote artifact: {path}")
        if type(entry.size) is not int or not 0 <= entry.size <= limits.max_file_bytes:
            raise ValueError("Remote artifact size exceeds bound")
        total += entry.size
        if total > limits.max_total_bytes:
            raise ValueError("Artifact download exceeds total byte bound")
        selected.append({"path": path, "name": relative.name, "bytes": entry.size})
    if not {"config.json", "status.json"} <= {row["name"] for row in selected}:
        raise ValueError("Missing run identity/completion metadata")
    return {"run_id": run_id, "files": selected, "remote_only": skipped, "bytes": total}


def receive_file(chunks, row: dict, target: Path, receipt: Path):
    """A failed stream remains a .part; only an exact complete stream is published."""
    if target.exists() or target.is_symlink():
        raise ValueError("Unreceipted existing artifact cannot be overwritten")
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    written = 0
    with partial.open("xb") as stream:
        for chunk in chunks:
            if type(chunk) is not bytes or written + len(chunk) > row["bytes"]:
                raise ValueError("Artifact stream exceeds inventory")
            stream.write(chunk)
            written += len(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    if written != row["bytes"]:
        raise ValueError("Artifact stream shorter than inventory")
    if target.suffix == ".json":
        read_json(partial)
    # Record the completed stream first: a crash between link and receipt is recoverable.
    write_json(receipt, {"file": row, "partial": str(partial), "complete_stream": True})
    os.link(partial, target)
    partial.unlink()


def _restore_file(row: dict, target: Path, receipt: Path) -> bool:
    if not receipt.exists():
        if target.exists() or target.is_symlink():
            raise ValueError("Unverified existing artifact; explicit reconciliation required")
        return False
    saved = read_json(receipt)
    if saved.get("file") != row or saved.get("complete_stream") is not True:
        raise ValueError("Artifact receipt mismatch")
    if not target.exists():
        partial = Path(saved["partial"])
        if (partial.parent != target.parent or partial.is_symlink() or not partial.is_file()
                or partial.stat().st_size != row["bytes"]):
            raise ValueError("Completed artifact publication cannot be recovered")
        os.link(partial, target)
        partial.unlink()
    if target.is_symlink() or target.stat().st_size != row["bytes"]:
        raise ValueError("Completed artifact length changed")
    if target.suffix == ".json":
        read_json(target)
    return True


def download_artifacts(volume_name: str, run_id: str, destination: Path, *, environment="main",
                       limits=None, worker=run_bounded) -> Path:
    """Resume completed files, retry only bounded reads, never recreate a compute call."""
    limits = limits or TransportLimits()
    safe_run_id(run_id)
    destination = Path(destination).resolve()
    stage = destination / ".transfers" / run_id
    target_dir = destination / "runs" / run_id
    stage.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    common = [sys.executable, "-m", "cloud_edge_rl.transport", "--volume", volume_name,
              "--run-id", run_id, "--environment", environment,
              "--limits", json.dumps(asdict(limits))]
    inventory_path = stage / "inventory.json"
    if not inventory_path.exists():
        worker([*common, "--action", "list", "--output", str(inventory_path)],
               min(limits.listing_seconds, limits.download_seconds),
               stage / f"list-{uuid.uuid4().hex}.log")
    inventory = read_json(inventory_path)
    if inventory.get("run_id") != run_id:
        raise ValueError("Download run identity changed")
    # Revalidate cached inventory against current bounds without contacting the cloud.
    from types import SimpleNamespace
    checked = inventory_entries([SimpleNamespace(path=row["path"], size=row["bytes"], type=1)
                                 for row in inventory["files"]], run_id, limits)
    if checked["files"] != inventory["files"] or checked["bytes"] != inventory["bytes"]:
        raise ValueError("Download inventory changed")
    for row in inventory["files"]:
        target = target_dir / row["name"]
        receipt = stage / (row["name"] + ".receipt.json")
        if _restore_file(row, target, receipt):
            continue
        for attempt in range(limits.read_attempts):
            remaining = limits.download_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Download stage deadline; completed files retained")
            try:
                worker([*common, "--action", "read", "--remote-path", row["path"],
                        "--bytes", str(row["bytes"]), "--output", str(target),
                        "--receipt", str(receipt)], min(limits.file_seconds, remaining),
                       stage / f"read-{row['name']}-{uuid.uuid4().hex}.log")
                if not _restore_file(row, target, receipt):
                    raise ValueError("Worker did not publish a completion receipt")
                break
            except TimeoutError:
                # Transport reads only. Deterministic/schema errors are not retried.
                if _restore_file(row, target, receipt):
                    break
                if attempt + 1 == limits.read_attempts:
                    raise
    config, status = read_json(target_dir / "config.json"), read_json(target_dir / "status.json")
    if config.get("run_id") != run_id or status.get("state") not in ("completed", "failed"):
        raise ValueError("Downloaded run identity or terminal state invalid")
    marker = stage / "complete.json"
    if not marker.exists():
        write_json(marker, {"run_id": run_id, "files": inventory["files"],
                            "transport_complete": True, "semantic_gate_required": True})
    return target_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("list", "read"), required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--remote-path")
    parser.add_argument("--bytes", type=int)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    limits = TransportLimits(**json.loads(args.limits))
    safe_run_id(args.run_id)
    import modal
    volume = modal.Volume.from_name(args.volume, environment_name=args.environment)
    if args.action == "list":
        inventory = inventory_entries(volume.iterdir("runs/" + args.run_id, recursive=False),
                                      args.run_id, limits)
        write_json(args.output, inventory)
    else:
        name = PurePosixPath(args.remote_path).name
        if (args.remote_path != "runs/" + args.run_id + "/" + name
                or name not in ARTIFACT_FILES or args.output.name != name
                or args.bytes is None or not 0 <= args.bytes <= limits.max_file_bytes
                or args.receipt is None):
            raise ValueError("Read worker requires an exact bounded artifact")
        receive_file(volume.read_file(args.remote_path),
                     {"name": name, "path": args.remote_path, "bytes": args.bytes},
                     args.output, args.receipt)


if __name__ == "__main__":
    main()
