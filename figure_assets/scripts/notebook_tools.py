"""Isolated notebook orchestration and leased, read-only results preview.

No original experiment module is imported. All writes are confined to the new
presentation directories; scientific settings, checkpoints and ledgers are read
only by the separate, allowlisted export adapter.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

import psutil


def project_root(root: Path | str | None = None) -> Path:
    result = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    if not (result / "figure_assets" / "scripts" / "export_results.py").is_file():
        raise ValueError("Select the RL_project root containing figure_assets/scripts/export_results.py")
    return result


def read_json(path: Path) -> dict:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"Presentation JSON exceeds the 32 MiB read limit: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _confined(base: Path, relative: str) -> Path:
    # 拒绝越出展示目录的路径。 / Reject paths outside the presentation directory.
    parts = PurePosixPath(relative.replace("\\", "/"))
    if parts.is_absolute() or not parts.parts or any(p in {".", ".."} or ":" in p for p in parts.parts):
        raise ValueError("Unconfined presentation path")
    result = base.joinpath(*parts.parts)
    current = base
    if base.is_symlink():
        raise ValueError("Symbolic presentation root is forbidden")
    for part in parts.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symbolic presentation path is forbidden")
    if not result.resolve().is_relative_to(base.resolve()):
        raise ValueError("Unconfined presentation path")
    return result


def open_workspace(root: Path | str | None = None) -> dict:
    """Reconstruct presentation state from disk, including after a kernel restart."""
    # 从磁盘恢复状态，不依赖旧内核变量。 / Restore disk state without depending on old kernel variables.
    root = project_root(root)
    index_path = root / "frontend/public/data/index.json"
    snapshot_path = None
    snapshot = None
    if index_path.is_file():
        index = read_json(index_path)
        if index.get("schema_version") != "results_index_v1":
            raise ValueError("Unsupported results index")
        entries = [r for r in index.get("snapshots", [])
                   if r.get("snapshot_id") == index.get("current_snapshot")]
        if len(entries) != 1:
            raise ValueError("The current snapshot is not uniquely registered")
        snapshot_path = _confined(index_path.parent, entries[0]["path"])
        snapshot = read_json(snapshot_path)
        if (snapshot.get("schema_version") != "results_visualization_v1"
                or snapshot.get("snapshot_id") != index["current_snapshot"]):
            raise ValueError("Snapshot/index identity mismatch")
    manifest_path = None
    manifest = None
    if snapshot is not None:
        manifest_path = _confined(root / "figure_assets", "manifest.json")
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if manifest.get("schema_version") != "figure_manifest_v1":
                raise ValueError("Unsupported figure manifest")
            if manifest.get("snapshot_id") != snapshot["snapshot_id"]:
                manifest = None  # 新快照尚未绘图。 / Figures for the new snapshot are not rendered yet.
    return {"root": root, "snapshot_path": snapshot_path, "snapshot": snapshot,
            "manifest_path": manifest_path, "manifest": manifest}


def _live(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _clean_tree(processes: list[psutil.Process]) -> list[int]:
    """Keep psutil's creation-time identities; a reused PID is not our child."""
    # 使用 PID 和创建时间识别本次子进程。 / Identify owned children by PID and creation time.
    unique = {}
    for process in processes:
        try:
            if _live(process):
                unique[(process.pid, process.create_time())] = process
        except psutil.Error:
            pass
    for process in unique.values():
        try:
            process.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(list(unique.values()), timeout=3)
    for process in alive:
        if _live(process):
            try:
                process.kill()
            except psutil.Error:
                pass
    psutil.wait_procs(alive, timeout=3)
    return [p.pid for p in unique.values() if _live(p)]


def run_stage(command: list[str], *, root: Path, timeout_seconds: int, name: str,
              cwd: Path | None = None) -> dict:
    """A bounded subprocess stage with a process-tree watchdog and durable receipt."""
    # 分阶段限时运行，并保存日志与清理结果。 / Bound each stage and persist logs and cleanup results.
    root = project_root(root)
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 300:
        raise ValueError("A stage timeout must be an integer between 1 and 300 seconds")
    if not name.replace("-", "").isalnum():
        raise ValueError("Invalid stage name")
    directory = root / "figure_assets/reports/stages"
    directory.mkdir(parents=True, exist_ok=True)
    stage_id = f"{name}-{uuid.uuid4().hex}"
    log_path = directory / f"{stage_id}.log"
    children = {}
    started = time.monotonic()
    timed_out = False
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    with log_path.open("wb") as log:
        child = subprocess.Popen(command, cwd=cwd or root, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, stdin=subprocess.DEVNULL)
        parent = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                for descendant in parent.children(recursive=True):
                    children[(descendant.pid, descendant.create_time())] = descendant
            except psutil.Error:
                pass
            if time.monotonic() - started >= timeout_seconds:
                timed_out = True
                break
            time.sleep(0.05)
        survivors = _clean_tree([*children.values(), parent] if timed_out else list(children.values()))
        try:
            exit_code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            exit_code = None
    record = {"stage_id": stage_id, "command": command, "exit_code": exit_code,
              "elapsed_seconds": round(time.monotonic() - started, 3), "timed_out": timed_out,
              "process_tree_clean": not survivors, "surviving_pids": survivors,
              "log_path": str(log_path), "new_cloud_calls": 0, "new_interactions": 0}
    _write_json(directory / f"{stage_id}.json", record)
    if timed_out or exit_code != 0 or survivors:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-2400:]
        raise RuntimeError(f"Presentation stage {name} failed; see {log_path}\n{detail}")
    return record


def export_data(root: Path | str | None = None, *, timeout_seconds: int = 60) -> dict:
    root = project_root(root)
    stage = run_stage([sys.executable, str(root / "figure_assets/scripts/export_results.py"),
                       "--root", str(root)], root=root, timeout_seconds=timeout_seconds, name="export")
    context = open_workspace(root)
    if context["snapshot"] is None:
        raise RuntimeError("Export returned successfully without registering a snapshot")
    return {"snapshot_id": context["snapshot"]["snapshot_id"],
            "data_path": str(context["snapshot_path"]), "stage": stage}


def generate_figures(root: Path | str | None = None, *, width: str = "double",
                     timeout_seconds: int = 90) -> dict:
    # 论文图片使用前端的同一份快照。 / Render paper figures from the frontend's shared snapshot.
    root = project_root(root)
    if width not in {"single", "double"}:
        raise ValueError("Choose single or double column width")
    context = open_workspace(root)
    if context["snapshot_path"] is None:
        raise RuntimeError("No result snapshot: run the export section first")
    stage = run_stage([sys.executable, str(root / "figure_assets/scripts/render_figures.py"),
                       "--root", str(root), "--snapshot", str(context["snapshot_path"]),
                       "--width", width], root=root, timeout_seconds=timeout_seconds, name="render")
    context = open_workspace(root)
    if context["manifest"] is None:
        raise RuntimeError("Renderer returned successfully without a figure manifest")
    return {"manifest_path": str(context["manifest_path"]), "stage": stage,
            "figures": context["manifest"]["figures"], "skipped": context["manifest"]["skipped"]}


def build_frontend(root: Path | str | None = None, *, timeout_seconds: int = 90) -> dict:
    # 仅本地构建，不自动安装依赖。 / Build locally without automatically installing dependencies.
    root = project_root(root)
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None or not (root / "frontend/node_modules").is_dir():
        raise RuntimeError("Install the frontend dependencies first: cd frontend, then npm ci")
    npm_cli = Path(npm).parent / "node_modules/npm/bin/npm-cli.js"
    node = shutil.which("node")
    command = [node, str(npm_cli), "run", "build"] if node and npm_cli.is_file() else [npm, "run", "build"]
    stage = run_stage(command, root=root, cwd=root / "frontend",
                      timeout_seconds=timeout_seconds, name="frontend-build")
    if not (root / "frontend/dist/index.html").is_file():
        raise RuntimeError("Frontend build produced no index.html")
    return stage


def _state_path(root: Path) -> Path:
    return root / "figure_assets/reports/preview-server.json"


def _bound_process(record: dict, root: Path) -> psutil.Process | None:
    # 核对进程身份，避免误停复用 PID 的其他进程。 / Verify identity to avoid stopping a reused PID.
    if record.get("schema_version") != "visualization_preview_v1" or record.get("root") != str(root):
        raise ValueError("The preview receipt belongs to a different workspace")
    try:
        process = psutil.Process(record["pid"])
        if abs(process.create_time() - record["create_time"]) > 0.001:
            raise ValueError("The recorded PID was reused; refusing to stop another process")
        command = process.cmdline()
        if (record["instance_id"] not in command or "--instance-id" not in command
                or "figure_assets.scripts.notebook_tools" not in command
                or str(root) not in command or "serve" not in command):
            raise ValueError("Preview process identity changed; refusing to stop another process")
        return process if _live(process) else None
    except psutil.NoSuchProcess:
        return None


def preview_status(root: Path | str | None = None) -> dict:
    root = project_root(root)
    path = _state_path(root)
    if not path.is_file():
        return {"status": "not_started"}
    record = read_json(path)
    process = _bound_process(record, root)
    if process is None:
        return {**record, "status": "stopped"}
    try:
        with urlopen(f"{record['url']}__preview__/health", timeout=2) as response:
            health = json.load(response)
        if health.get("instance_id") != record["instance_id"]:
            raise ValueError("Preview endpoint identity does not match its receipt")
    except (OSError, URLError) as error:
        return {**record, "status": "unreachable", "error": str(error)}
    return {**record, "status": "running", "remaining_seconds": health["remaining_seconds"]}


def start_preview(root: Path | str | None = None, *, lease_seconds: int = 1800,
                  port: int = 0) -> dict:
    """Start once, or reconnect to the same live lease; never starts a cloud job."""
    # 复用存活预览，不延长原有期限。 / Reuse a live preview without renewing its lease.
    root = project_root(root)
    if type(lease_seconds) is not int or not 2 <= lease_seconds <= 3600:
        raise ValueError("Preview lease must be between 2 and 3600 seconds")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Invalid loopback port")
    old = preview_status(root)
    if old["status"] == "running":
        return old
    if old["status"] == "unreachable":
        raise RuntimeError("An existing preview process is unreachable; stop that exact lease first")
    if not (root / "frontend/dist/index.html").is_file():
        raise RuntimeError("Build the frontend before starting its preview")
    instance_id = uuid.uuid4().hex
    log_path = root / f"figure_assets/reports/preview-{instance_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        child = subprocess.Popen([sys.executable, "-m", "figure_assets.scripts.notebook_tools", "serve",
                                  "--root", str(root), "--port", str(port), "--lease-seconds", str(lease_seconds),
                                  "--instance-id", instance_id], cwd=root, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONIOENCODING="utf-8",
                                          PYTHONPATH=os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[2]),
                                                                                   os.environ.get("PYTHONPATH")]))))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Preview exited before readiness: {log_path.read_text(encoding='utf-8')}")
        if _state_path(root).is_file():
            record = read_json(_state_path(root))
            if record.get("instance_id") == instance_id:
                status = preview_status(root)
                if status["status"] == "running":
                    return status
        time.sleep(0.1)
    survivors = _clean_tree([psutil.Process(child.pid)])
    raise RuntimeError(f"Preview readiness deadline exceeded: {log_path}; surviving PIDs: {survivors}")


def stop_preview(root: Path | str | None = None) -> dict:
    """Stop only the process matching its UUID, root and creation-time receipt."""
    # 只停止当前工作区绑定的本地预览。 / Stop only this workspace's bound local preview.
    root = project_root(root)
    path = _state_path(root)
    if not path.is_file():
        return {"status": "not_started"}
    record = read_json(path)
    process = _bound_process(record, root)
    if process is None:
        return {**record, "status": "stopped", "process_tree_clean": True}
    try:
        request = Request(f"{record['url']}__preview__/stop", method="POST",
                          headers={"X-Preview-Instance": record["instance_id"]})
        with urlopen(request, timeout=2) as response:
            response.read()
        process.wait(timeout=4)
    except (OSError, URLError, psutil.TimeoutExpired):
        survivors = _clean_tree([*process.children(recursive=True), process])
        if survivors:
            raise RuntimeError(f"Preview process-tree cleanup failed: {survivors}")
    result = {**record, "status": "stopped", "process_tree_clean": True}
    _write_json(path, result)
    return result


def _serve(root: Path, *, port: int, lease_seconds: int, instance_id: str) -> None:
    """Only public snapshots, built UI, and generated figures are addressable."""
    # 仅开放展示文件，不提供实验执行接口。 / Serve presentation files without an experiment execution API.
    stop = threading.Event()
    deadline = time.monotonic() + lease_seconds
    state = {}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def _json(self, value, status=200):
            body = json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if (urlsplit(self.path).path == "/__preview__/stop"
                    and self.headers.get("X-Preview-Instance") == instance_id):
                self._json({"status": "stopping"})
                stop.set()
            else:
                self._json({"error": "Read-only preview; no execution API"}, 405)

        def do_HEAD(self):
            self.do_GET(head=True)

        def do_GET(self, head=False):
            path = unquote(urlsplit(self.path).path)
            if path == "/__preview__/health":
                self._json({"instance_id": instance_id,
                            "remaining_seconds": max(0, round(deadline - time.monotonic()))})
                return
            try:
                if path.startswith("/figure-assets/"):
                    base, relative = root / "figure_assets", path[len("/figure-assets/"):]
                    if not (relative == "manifest.json"
                            or (relative.startswith("main/") and relative.endswith(".png"))
                            or (relative.startswith("reports/figure-data/") and relative.endswith(".json"))):
                        self._json({"error": "Public artifact not found"}, 404)
                        return
                elif path.startswith("/data/"):
                    base, relative = root / "frontend/public/data", path[len("/data/"):]
                else:
                    base, relative = root / "frontend/dist", path.lstrip("/") or "index.html"
                target = _confined(base, relative)
                if not target.is_file() and base == root / "frontend/dist" and not target.suffix:
                    target = base / "index.html"
                if not target.is_file() or target.stat().st_size > 32 * 1024 * 1024:
                    self._json({"error": "Public artifact not found"}, 404)
                    return
                content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(target.stat().st_size))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                if not head:
                    with target.open("rb") as source:
                        shutil.copyfileobj(source, self.wfile, length=64 * 1024)
            except ValueError:
                self._json({"error": "Forbidden presentation path"}, 403)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False

    with Server(("127.0.0.1", port), Handler) as server:
        server.timeout = 0.2
        process = psutil.Process()
        state.update(schema_version="visualization_preview_v1", instance_id=instance_id,
                     root=str(root), pid=process.pid, create_time=process.create_time(), status="running",
                     url=f"http://127.0.0.1:{server.server_port}/", lease_seconds=lease_seconds,
                     started_at=datetime.now(timezone.utc).isoformat(), process_tree_clean=None)
        _write_json(_state_path(root), state)
        try:
            while not stop.is_set() and time.monotonic() < deadline:
                server.handle_request()
                # 异常子进程触发自看门狗清理。 / Unexpected children trigger self-watchdog cleanup.
                children = process.children(recursive=True)
                if children:
                    survivors = _clean_tree(children)
                    state["unexpected_child_pids"] = [p.pid for p in children]
                    state["surviving_pids"] = survivors
                    break
        finally:
            survivors = _clean_tree(process.children(recursive=True))
            state.update(status="stopped" if stop.is_set() else "expired",
                         process_tree_clean=not survivors, surviving_pids=survivors)
            _write_json(_state_path(root), state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["serve", "status", "stop"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--lease-seconds", type=int, default=1800)
    parser.add_argument("--instance-id")
    args = parser.parse_args()
    root = project_root(args.root)
    if args.action == "serve":
        if not args.instance_id or not 2 <= args.lease_seconds <= 3600 or not 0 <= args.port <= 65535:
            parser.error("Serve requires an instance ID and a bounded loopback lease")
        _serve(root, port=args.port, lease_seconds=args.lease_seconds, instance_id=args.instance_id)
    else:
        print(json.dumps(preview_status(root) if args.action == "status" else stop_preview(root), indent=2))


if __name__ == "__main__":
    main()
