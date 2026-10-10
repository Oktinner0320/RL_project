"""One invocation for frozen training jobs; durable barriers and no compute retries."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

from cloud_edge_rl.config import RunSpec
from cloud_edge_rl.transport import (
    POLICY_SNAPSHOTS,
    TransportLimits,
    download_artifacts,
    download_dependency_artifacts,
    download_public_updates,
    read_json,
    run_bounded,
    safe_run_id,
    stop_and_verify,
    write_json,
)


def _same(left, right):
    from cloud_edge_rl.comparisons.acceptance import _first_json_equal
    return _first_json_equal(left, right)


def _state_equal(left, right):
    import numpy as np
    import torch
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right))
    return left == right


def _limits(spec):
    return TransportLimits(**spec.config["modal"].get("transport", {}))


@contextmanager
def _controller_lock(directory: Path):
    import psutil
    path = directory / "controller.lock.json"
    token = uuid.uuid4().hex
    for _ in range(2):
        try:
            with path.open("x", encoding="utf-8") as stream:
                import json
                json.dump({"pid": os.getpid(), "created": psutil.Process().create_time(),
                           "token": token}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            break
        except FileExistsError:
            old = read_json(path)
            try:
                process = psutil.Process(old["pid"])
                if process.create_time() == old["created"]:
                    raise ValueError("Training controller already running for this output")
            except psutil.NoSuchProcess:
                pass
            try:
                path.rename(directory / f"stale-controller-{uuid.uuid4().hex}.json")
            except FileNotFoundError:
                continue
    else:
        raise ValueError("Concurrent training controller lock acquisition")
    try:
        yield
    finally:
        if read_json(path).get("token") != token:
            raise ValueError("Training controller lock owner changed")
        path.unlink()


def _compute_receipt(directory: Path, spec: RunSpec):
    path = directory / "compute-result.json"
    if not path.exists():
        return None
    saved = read_json(path)
    binding = read_json(directory / "call.json")
    if (saved.get("run_id") != spec.run_id or saved.get("call_id") != binding.get("call_id")
            or saved.get("app_id") != binding.get("app_id")
            or binding.get("run_id") != spec.run_id or saved.get("compute_completed") is not True
            or type(saved.get("result")) is not dict
            or saved["result"].get("run_id") != spec.run_id):
        raise ValueError("Completed compute receipt/binding mismatch")
    return saved


def run_training_pipeline(specs: list[RunSpec], root: Path, directory: Path, *,
                          verify="cloud_edge_rl.training_pipeline:verify_training_artifacts",
                          worker=run_bounded, downloader=download_artifacts,
                          artifact_destination: Path | None = None,
                          transport_limits: TransportLimits | None = None,
                          local_only: bool = False) -> dict:
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with _controller_lock(directory):
        return _run_training_pipeline(specs, root, directory, verify=verify,
                                      worker=worker, downloader=downloader,
                                      artifact_destination=artifact_destination,
                                      transport_limits=transport_limits, local_only=local_only)


def _validate_controller_specs(specs, root, verify):
    """Only an exact admitted fresh scope widens the historical native route."""
    from cloud_edge_rl.comparisons.acceptance import _first_artifact
    purpose = 'fresh_seven_method_development_benchmark_v1'
    markers = [spec.config.get('native_acceptance', {}).get('fresh_benchmark') for spec in specs]
    if any(marker is not None for marker in markers):
        if verify != 'cloud_edge_rl.fresh_benchmark:verify_fresh_artifacts':
            raise ValueError('Fresh scope requires its original semantic verifier')
        routes = {'collect': {'round_robin'}, 'benchmark_attack': {'gradient_matching'},
                  'audit': {'gradient_matching'}, 'fit_risk': {'risk_ensemble'},
                  'train_policy': {'dqn', 'ppo', 'pqn'},
                  'validate': {'dqn', 'ppo', 'pqn', 'metacs', 'fixed', 'time_greedy', 'risk_greedy'}}
        for spec, marker in zip(specs, markers):
            descriptor = spec.config.get('native_acceptance', {}).get('execution_admission')
            if (type(marker) is not dict or marker.get('schema') != purpose
                    or descriptor is None or spec.method not in routes.get(spec.stage, set())):
                raise ValueError('Registered fresh scope and mandatory admission required')
            witness = _first_artifact(descriptor)
            if witness.get('purpose') != purpose or witness.get('approved') is not True:
                raise ValueError('Only exact approved fresh purpose can generalize controller')
        from cloud_edge_rl.admission import check_admission
        check_admission(specs, root)
    elif any(spec.stage != 'train_policy' or spec.method not in ('dqn', 'ppo') for spec in specs):
        raise ValueError('This controller accepts registered native DQN/PPO training jobs only')


def _run_training_pipeline(specs: list[RunSpec], root: Path, directory: Path, *,
                          verify="cloud_edge_rl.training_pipeline:verify_training_artifacts",
                          worker=run_bounded, downloader=download_artifacts,
                          artifact_destination: Path | None = None,
                           transport_limits: TransportLimits | None = None,
                           local_only: bool = False) -> dict:
    """Run approved jobs serially. `verify` is a required, local semantic gate.

    Existing completed receipts survive cleanup/download failures. Unknown spawns remain
    reserved. An invocation may reattach a known call, but can never resubmit that job.
    Full training targets stay in the frozen specs, not in this transport controller.
    """
    if (not specs or len(specs) > 32 or type(verify) is not str
            or not verify.startswith("cloud_edge_rl.") or verify.count(":") != 1):
        raise ValueError("1..32 frozen jobs and an explicit semantic gate required")
    if type(local_only) is not bool:
        raise ValueError('Explicit local-only mode must be boolean')
    _validate_controller_specs(specs, root, verify)
    if len({spec.run_id for spec in specs}) != len(specs):
        raise ValueError("Duplicate registered run ID")
    if transport_limits is not None and type(transport_limits) is not TransportLimits:
        raise ValueError("Explicit bounded transport limits required")
    root, directory = Path(root).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # The immutable manifest is an execution snapshot, not a second CONFIG entry point.
    artifacts_root = (directory / "artifacts" if artifact_destination is None
                      else Path(artifact_destination).resolve())
    manifest = {"run_ids": [safe_run_id(spec.run_id) for spec in specs],
                "artifact_destination": str(artifacts_root), "verifier": verify,
                "transport_override": None if transport_limits is None else asdict(transport_limits)}
    manifest_path = directory / "manifest.json"
    if local_only and not manifest_path.exists():
        raise ValueError('Local reverification requires original controller manifest')
    if manifest_path.exists():
        if not _same(read_json(manifest_path), manifest):
            raise ValueError("Training manifest cannot change on resume")
    else:
        write_json(manifest_path, manifest)
    state_path = directory / "state.json"
    if local_only and not state_path.exists():
        raise ValueError('Local reverification requires original controller state')
    state = read_json(state_path) if state_path.exists() else {"jobs": {}, "status": "ready"}
    if set(state["jobs"]) - set(manifest["run_ids"]):
        raise ValueError("Pipeline state contains an unregistered job")
    fresh = []
    for spec in specs:
        limits = _limits(spec) if transport_limits is None else transport_limits
        if (spec.config["execution"]["backend"] != "modal"
                or any(type(spec.config["modal"][key]) is not int
                       or spec.config["modal"][key] <= 0 for key in
                       ("timeout_seconds", "startup_timeout_seconds"))):
            raise ValueError("Explicit positive cloud stage deadlines required")
        stage = directory / "jobs" / spec.run_id
        stage.mkdir(parents=True, exist_ok=True)
        frozen = stage / "spec.json"
        if local_only and (not frozen.exists() or not (stage / 'transport.json').exists()):
            raise ValueError('Local reverification requires original frozen spec and transport')
        if frozen.exists():
            if not _same(read_json(frozen), asdict(spec)):
                raise ValueError("Frozen training spec changed on resume")
        else:
            write_json(frozen, asdict(spec))
        transport_path = stage / "transport.json"
        if transport_path.exists():
            if not _same(read_json(transport_path), asdict(limits)):
                raise ValueError("Frozen transport limits changed on resume")
        else:
            write_json(transport_path, asdict(limits))
        if not (stage / "call.json").exists():
            fresh.append(spec)
    if fresh:
        if local_only:
            raise ValueError('Local reverification cannot submit or recover unknown compute')
        from cloud_edge_rl.admission import check_admission
        check_admission(fresh, root)
    try:
        for spec in specs:
            limits = _limits(spec) if transport_limits is None else transport_limits
            stage = directory / "jobs" / spec.run_id
            stage.mkdir(parents=True, exist_ok=True)
            row = state["jobs"].setdefault(spec.run_id, {})
            common = [sys.executable, "-m", "cloud_edge_rl.training_pipeline",
                      "--root", str(root), "--stage", str(stage)]
            receipt = _compute_receipt(stage, spec)
            if local_only and receipt is None:
                raise ValueError('Local reverification requires a completed bound compute receipt')
            if local_only:
                from cloud_edge_rl.admission import bound_call
                if bound_call(spec, root) != receipt['call_id']:
                    raise ValueError('Local reverification paid journal binding mismatch')
            if receipt is None:
                timeout = (spec.config["modal"]["timeout_seconds"]
                           + spec.config["modal"]["startup_timeout_seconds"]
                           + limits.cleanup_seconds)
                row["phase"] = "compute"
                write_json(state_path, state, replace=True)
                try:
                    worker([*common, "--action", "compute"], timeout,
                           stage / f"compute-{uuid.uuid4().hex}.log")
                except Exception as error:
                    row["controller_error"] = {"type": type(error).__name__, "message": str(error)}
                    # A completed GPU task is not made incomplete by an app context failure.
                    receipt = _compute_receipt(stage, spec)
                    if receipt is None:
                        raise
                receipt = _compute_receipt(stage, spec)
                if receipt is None:
                    raise ValueError("Compute worker did not persist a completed return")
            row.update(compute_completed=True, call_id=receipt["call_id"], app_id=receipt["app_id"])
            row["phase"] = "cleanup"
            write_json(state_path, state, replace=True)
            shutdown = stage / "shutdown.json"
            if not shutdown.exists():
                if local_only:
                    raise ValueError('Local reverification requires retained exact shutdown proof')
                worker([*common, "--action", "cleanup"], limits.cleanup_seconds + 10,
                       stage / f"cleanup-{uuid.uuid4().hex}.log")
            proof = read_json(shutdown)
            if (proof.get("app_id") != receipt["app_id"] or proof.get("verified") is not True
                    or proof.get("stopped") is not True):
                raise ValueError("Exact app shutdown proof required before advancing")
            row.update(shutdown_verified=True, phase="download")
            write_json(state_path, state, replace=True)
            if local_only:
                artifacts = Path(row.get('artifact_directory', artifacts_root / 'runs' / spec.run_id)).resolve()
                if not artifacts.is_relative_to(artifacts_root.resolve()) or not artifacts.is_dir():
                    raise ValueError('Local reverification requires confined cached artifacts')
            else:
                fresh_scope = verify == 'cloud_edge_rl.fresh_benchmark:verify_fresh_artifacts'
                extra = {}
                if fresh_scope and spec.stage == 'train_policy':
                    steps = spec.config['native_acceptance']['fresh_benchmark']['checkpoint_steps']
                    if steps != [500, 1000, 1500, 2000, 2500, 3000]:
                        raise ValueError('Exact registered fresh curve snapshots required')
                    extra['policy_snapshots'] = POLICY_SNAPSHOTS
                    limits = replace(limits, max_entries=64)
                artifacts = downloader(spec.config["modal"]["volume_name"], spec.run_id,
                                       artifacts_root, limits=limits, **extra)
                row['artifact_directory'] = str(artifacts)
                write_json(state_path, state, replace=True)
                download_dependency_artifacts(spec, artifacts_root, worker=worker, limits=limits)
                if fresh_scope and spec.stage == 'collect':
                    download_public_updates(spec, artifacts, worker=worker,
                                            limits=replace(limits, read_attempts=1, max_entries=66,
                                                           download_seconds=min(limits.download_seconds, 180)))
            row["phase"] = "verify"
            write_json(state_path, state, replace=True)
            # Re-run the semantic gate on resumed files: a receipt is NOT integrity proof.
            verification = stage / f"verification-{uuid.uuid4().hex}.json"
            worker([*common, "--action", "verify", "--verifier", verify,
                    "--artifacts", str(artifacts), "--verification-output", str(verification)],
                   limits.verification_seconds, stage / f"verify-{uuid.uuid4().hex}.log")
            proof = read_json(verification)
            if type(proof) is not dict or proof.get("verified") is not True:
                raise ValueError("Training semantic gate did not verify the run")
            row.update(phase="verified", verification=proof, artifact_directory=str(artifacts))
            write_json(state_path, state, replace=True)
        state["status"] = "completed"
        write_json(state_path, state, replace=True)
        return state
    except BaseException as error:
        # A failed/timed-out compute can still have opened a remote app. Do not leave
        # it alive merely because no completed GPU return was published.
        if ("row" in locals() and row.get("phase") == "compute" and any((stage / name).exists()
                                                  for name in ("app.json", "call.json"))):
            try:
                worker([*common, "--action", "cleanup"], limits.cleanup_seconds + 10,
                       stage / f"failed-compute-cleanup-{uuid.uuid4().hex}.log")
                proof = read_json(stage / "shutdown.json")
                binding = read_json(stage / ("call.json" if (stage / "call.json").exists() else "app.json"))
                row["shutdown_verified"] = (proof.get("app_id") == binding.get("app_id")
                                            and proof.get("verified") is True and proof.get("stopped") is True)
            except BaseException as cleanup_error:
                row["shutdown_verified"] = False
                row["cleanup_error"] = {"type": type(cleanup_error).__name__,
                                        "message": str(cleanup_error)}
        state.update(status="blocked", error={"type": type(error).__name__, "message": str(error)})
        write_json(state_path, state, replace=True)
        raise


def execute_compute(spec: RunSpec, root: Path, directory: Path):
    """Persist the paid binding and return before leaving the app context."""
    import modal

    from cloud_edge_rl import admission, modal_runner

    call_path = directory / "call.json"
    wait_seconds = (spec.config["modal"]["timeout_seconds"]
                    + spec.config["modal"]["startup_timeout_seconds"])
    if call_path.exists():
        binding = read_json(call_path)
        if (binding.get("run_id") != spec.run_id
                or admission.bound_call(spec, root) != binding.get("call_id")):
            raise ValueError("Retained paid binding mismatch")
        # Read-only reattachment uses the original spec/binding, not a new source approval.
        call = modal.FunctionCall.from_id(binding["call_id"])
        result = call.get(timeout=wait_seconds)
        write_json(directory / "compute-result.json", {**binding, "result": result,
                                                       "compute_completed": True})
        return
    # All guards, including historical quotas/source/owner, precede any app creation.
    admission.check_admission([spec], root)
    bound = admission.reserve(spec, root)
    if bound is not None:
        raise ValueError("Paid binding exists without controller receipt; reconcile, do not spawn")
    app, function, _ = modal_runner.create_app(root, spec.config["modal"], [spec])
    call = None
    try:
        with app.run(detach=False, environment_name="main"):
            write_json(directory / "app.json", {"run_id": spec.run_id, "app_id": app.app_id})
            call = function.spawn(asdict(spec), spec.config["modal"]["volume_name"])
            admission.bind_call(spec, root, call.object_id)
            binding = {"run_id": spec.run_id, "app_id": app.app_id, "call_id": call.object_id}
            write_json(call_path, binding)
            result = call.get(timeout=wait_seconds)
            write_json(directory / "compute-result.json", {**binding, "result": result,
                                                           "compute_completed": True})
    except BaseException:
        if call is not None and not (directory / "compute-result.json").exists():
            call.cancel(terminate_containers=True)
        raise


def _cleanup_binding(spec: RunSpec, root: Path, directory: Path) -> dict:
    """Select only this job's retained app, including an interrupted unknown spawn."""
    from cloud_edge_rl import admission
    call_path = directory / "call.json"
    binding = read_json(call_path if call_path.exists() else directory / "app.json")
    if binding.get("run_id") != spec.run_id:
        raise ValueError("Cleanup app identity differs from registered run")
    if call_path.exists():
        if admission.bound_call(spec, root) != binding.get("call_id"):
            raise ValueError("Cleanup paid binding mismatch")
        if (directory / "app.json").exists():
            app = read_json(directory / "app.json")
            if app.get("run_id") != spec.run_id or app.get("app_id") != binding.get("app_id"):
                raise ValueError("Cleanup app and call receipts disagree")
    else:
        # Cleanup only: source edits may not reset or authorize the unknown reservation.
        admission.check_admission([spec], root, _saved_source=True)
    return binding


def verify_training_artifacts(spec: RunSpec, artifacts: Path, result: dict) -> dict:
    """Verify native checkpoint/counters/isolation, not utility or privacy acceptance."""
    import csv
    import math

    import torch

    from cloud_edge_rl.algorithms.policy import _settings
    from cloud_edge_rl.algorithms.resource_feedback import ResourceFeedback
    from cloud_edge_rl.artifacts import artifact_payload

    if not _same(artifact_payload(read_json(artifacts / "config.json")), artifact_payload(spec)):
        raise ValueError("Actual config/source differs from frozen training spec")
    if read_json(artifacts / "status.json").get("state") != "completed":
        raise ValueError("Training did not complete")
    if read_json(artifacts / "summary.json").get("run_id") != spec.run_id:
        raise ValueError("Training summary identity mismatch")
    runtime = read_json(artifacts / "runtime.json")
    requested_gpu = spec.config["modal"]["gpu"]
    if requested_gpu and (runtime.get("cuda_available") is not True
                          or requested_gpu not in runtime.get("gpu", "")):
        raise ValueError("Requested training GPU not observed")
    data = read_json(artifacts / "data_metadata.json")
    if spec.config["execution"]["split"] != "train":
        raise ValueError("Development training must not use sealed-test execution")
    if data["official_test_included"] is not False or data["rows"]["official_test"] != 0:
        raise ValueError("Official test entered development training")
    split = data["split"]
    train, validation = split["train_source_indices"], split["validation_source_indices"]
    clients = split["client_indices"]
    if (not _same(split["config"], spec.config["data"])
            or len(set(train)) != len(train) or len(set(validation)) != len(validation)
            or len(train) != data["rows"]["train"]
            or len(validation) != data["rows"]["validation"] or set(train) & set(validation)
            or sum(map(len, clients)) != len(train)
            or set().union(*map(set, clients)) != set(range(len(train)))):
        raise ValueError("Training/validation/private-client isolation mismatch")
    with (artifacts / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({(row["episode"], row["round"]) for row in rows}) != len(rows):
        raise ValueError("Training metric rows missing or duplicated")
    if any(row["risk_version"] != spec.risk_version or any(
            not math.isfinite(float(row[key])) for key in (
                "elapsed_seconds", "energy_joules", "validation_accuracy", "validation_loss", "reward"
            )) for row in rows):
        raise ValueError("Training metrics have invalid values or risk version")
    outer = torch.load(artifacts / "training.pt", map_location="cpu", weights_only=False)
    state = outer["state"]
    effective = {**spec.config["rl"], "seed": spec.seed, "algorithm": spec.method,
                 "risk_version": spec.risk_version,
                 "warmup_actions": spec.config["execution"]["warmup_actions"]}
    if (outer["run_id"] != spec.run_id or outer["risk_version"] != spec.risk_version
            or state["format"] != "native_training_v1"
            or state["risk_version"] != spec.risk_version
            or not _same(state["settings"], _settings(effective))):
        raise ValueError("Native training checkpoint identity/settings mismatch")
    previous = spec.inputs.get("previous_training_checkpoint", "")
    before = 0
    if previous:
        from pathlib import PurePosixPath
        remote = PurePosixPath(previous)
        if (len(remote.parts) != 3 or remote.parts[0] != "runs" or remote.name != "training.pt"
                or previous != remote.as_posix() or '\\' in previous):
            raise ValueError("Exact retained training checkpoint path required")
        safe_run_id(remote.parts[1])
        prior = torch.load(artifacts.parent / remote.parts[1] / remote.name,
                           map_location="cpu", weights_only=False)
        prior_state = prior['state']
        if (prior.get('run_id') != remote.parts[1] or prior.get('risk_version') != spec.risk_version
                or prior_state.get('format') != 'native_training_v1'
                or prior_state.get('risk_version') != spec.risk_version
                or not _same(prior_state.get('settings'), _settings(effective))
                or any(type(prior_state.get(key)) is not int or prior_state[key] < 0 for key in
                       ('environment_steps', 'learned_steps', 'warmup_steps', 'updates'))
                or prior_state['environment_steps'] != prior_state['learned_steps'] + prior_state['warmup_steps']):
            raise ValueError('Retained training checkpoint identity/settings/counters mismatch')
        before = prior["state"]["environment_steps"]
    steps = state["environment_steps"]
    expected = min(spec.config["rl"]["total_steps"], before + spec.config["rl"].get(
        "stage_interaction_limit", spec.config["rl"]["total_steps"]))
    if (any(type(state[key]) is not int or state[key] < 0 for key in
            ("environment_steps", "learned_steps", "warmup_steps", "updates"))
            or steps != state["learned_steps"] + state["warmup_steps"]
            or steps != expected or len(rows) != steps - before
            or type(result.get('learned_steps')) is not int
            or type(result.get('warmup_steps')) is not int
            or result.get("learned_steps") != state["learned_steps"]
            or result.get("warmup_steps") != state["warmup_steps"]):
        raise ValueError("Actual training counters differ from registered target")
    if spec.config["rl"].get("resource_feedback") is not None:
        if state["resource_feedback_total_steps"] != spec.config["rl"]["total_steps"]:
            raise ValueError("Resource feedback lifetime budget changed")
        feedback = ResourceFeedback(spec.config["rl"]["resource_feedback"], spec.config["reward"])
        feedback.load_snapshot(state["resource_feedback"])
    if any(not torch.isfinite(value).all().item() for value in state["agent"]["model"].values()):
        raise ValueError("Nonfinite trained model")
    agent = torch.load(artifacts / "agent.pt", map_location="cpu", weights_only=False)
    if (agent["risk_version"] != spec.risk_version
            or not _state_equal(agent["training"], state)
            or not _state_equal(agent["agent"], state["agent"])):
        raise ValueError("Agent and training checkpoints disagree")
    return {"verified": True, "run_id": spec.run_id, "actual_interactions": len(rows),
            "cumulative_interactions": steps, "optimizer_updates": state["updates"],
            "source_verified": True, "isolation_verified": True,
            "formal_acceptance": "not_evaluated", "actual_cloud_cost_usd": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("run", "compute", "cleanup", "verify"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--spec", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--artifact-destination", type=Path)
    parser.add_argument("--verification-output", type=Path)
    parser.add_argument("--verifier", default="cloud_edge_rl.training_pipeline:verify_training_artifacts")
    args = parser.parse_args()
    if args.action == "run":
        if not args.spec or args.output is None:
            parser.error("run needs registered --spec paths and --output")
        run_training_pipeline([RunSpec(**read_json(path)) for path in args.spec], args.root,
                              args.output, verify=args.verifier,
                              artifact_destination=args.artifact_destination)
        return
    if args.stage is None:
        parser.error("worker needs --stage")
    spec = RunSpec(**read_json(args.stage / "spec.json"))
    if args.action == "compute":
        execute_compute(spec, args.root, args.stage)
    elif args.action == "cleanup":
        binding = _cleanup_binding(spec, args.root, args.stage)
        transport_path = args.stage / "transport.json"
        limits = TransportLimits(**read_json(transport_path)) if transport_path.exists() else _limits(spec)
        proof = stop_and_verify(binding["app_id"], limits=limits)
        write_json(args.stage / "shutdown.json", proof)
    else:
        import importlib
        if (not args.verifier.startswith("cloud_edge_rl.") or args.verifier.count(":") != 1
                or args.artifacts is None or args.verification_output is None):
            raise ValueError("Explicit bounded semantic verifier required")
        module, function = args.verifier.split(":")
        receipt = _compute_receipt(args.stage, spec)
        if receipt is None:
            raise ValueError("Verification needs a completed compute receipt")
        proof = getattr(importlib.import_module(module), function)(
            spec, args.artifacts, receipt["result"])
        write_json(args.verification_output, proof)


if __name__ == "__main__":
    main()
