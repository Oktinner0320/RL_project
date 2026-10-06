"""下载固定版本的官方参考代码。 / Download pinned official reference repositories."""

import hashlib
import io
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPOSITORIES = {
    "tabular_gia": "scaleoutsystems/tabular-GIA",
    "tabm": "yandex-research/tabm",
    "conformalopt": "chrismohrii/conformalopt",
    "ocp_unlock": "trustml-lab/ocp-semi-bandit",
    "pqn": "mttga/purejaxql",
    "metacs_fl": "alan-lira/metacs-fl",
}


def read_url(url: str) -> bytes:
    """读取公开源，保留上游错误。 / Read public sources without suppressing upstream errors."""
    request = urllib.request.Request(url, headers={"User-Agent": "cloud-edge-rl-reference-study"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def download_one(item: tuple[str, str], root: Path) -> dict:
    """固定 commit 并校验 ZIP 路径。 / Pin a commit and constrain archive paths."""
    name, slug = item
    target = root / "repos" / name
    record_path = root / "provenance" / f"{name}.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if target.exists():
            return record
        # 克隆项目后恢复同一版本。 / Restore the recorded revision after cloning the project.
        archive = read_url(record["archive_url"])
        if hashlib.sha256(archive).hexdigest() != record["archive_sha256"]:
            raise ValueError(f"Archive checksum differs from the recorded source: {name}")
    else:
        info = json.loads(read_url(f"https://api.github.com/repos/{slug}"))
        revision = "v0.2.0" if name == "metacs_fl" else info["default_branch"]
        commit = json.loads(read_url(f"https://api.github.com/repos/{slug}/commits/{revision}"))
        sha = commit["sha"]
        archive_url = f"https://codeload.github.com/{slug}/zip/{sha}"
        archive = read_url(archive_url)
        record = {
            "name": name,
            "repository": f"https://github.com/{slug}",
            "commit": sha,
            "revision": revision,
            "commit_date": commit["commit"]["committer"]["date"],
            "license_api": info.get("license"),
            "archive_url": archive_url,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "archive_bytes": len(archive),
            "download_date": "2026-10-05",
            "source_modified": False,
            "local_path": f"reference_algorithms/repos/{name}",
        }
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for entry in zipped.infolist():
            relative = Path(*Path(entry.filename).parts[1:])
            destination = target / relative
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError(f"Archive path escapes destination: {entry.filename}")
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(zipped.read(entry))
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: {record['commit'][:12]} ({len(archive):,} bytes)", flush=True)
    return record


def download_repositories(root: Path) -> list[dict]:
    """下载六种候选，不执行第三方安装脚本。 / Download six candidates without installer scripts."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda item: download_one(item, root), REPOSITORIES.items()))
    (root / "catalog.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    download_repositories(Path(__file__).resolve().parent)
