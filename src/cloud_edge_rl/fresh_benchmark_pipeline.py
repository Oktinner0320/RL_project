"""Bounded driver of owner-frozen fresh stages; never produces jobs or approval."""

import argparse
from dataclasses import asdict
import importlib
import hashlib
import json
from pathlib import Path

from .admission import check_admission
from .config import RunSpec
from .fresh_benchmark import build_plan
from .parameter_selection_pipeline import load_notebook_config
from .training_pipeline import _controller_lock, run_training_pipeline
from .transport import safe_run_id, write_json


EXPERIMENT = 'fresh-seven-methods-20261010-v1'
PURPOSE = 'fresh_seven_method_development_benchmark_v1'
VERIFIER = 'cloud_edge_rl.fresh_benchmark:verify_fresh_artifacts'


def _strict_json(path, budget):
    path = Path(path)
    size = path.stat().st_size
    if size > 64 * 1024 * 1024 or budget[0] + size > 128 * 1024 * 1024 or budget[1] >= 256:
        raise ValueError('Fresh stage JSON size/count limit exceeded')
    budget[0] += size
    budget[1] += 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)

    result = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                        parse_constant=constant, parse_float=_finite_float)
    return result


def _finite_float(value):
    import math
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite JSON number')
    return result


def _absolute(value):
    if type(value) is not str or not Path(value).is_absolute():
        raise ValueError('Absolute frozen artifact path required')
    return Path(value).resolve()


def _gate(proof, run_id):
    if (type(proof) is not dict or proof.get('verified') is not True
            or proof.get('experiment_id') != EXPERIMENT or proof.get('run_id') != run_id
            or type(proof.get('physical_interactions')) is not int
            or proof['physical_interactions'] < 0):
        raise ValueError('Fresh semantic gate with physical interaction counter required')
    return proof['physical_interactions']


def _journal_identity(descriptor, directory, specs):
    """Bind external frozen descriptors without copying their nested payloads."""
    encoded = json.dumps(descriptor, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return {'schema': 'fresh_benchmark_journal_v2',
            'descriptor_ref': str(directory / 'stage.json'),
            'descriptor_sha256': hashlib.sha256(encoded).hexdigest(),
            'freeze_id': descriptor['freeze_id'],
            'spec_refs': list(descriptor['spec_paths']),
            'run_ids': [spec.run_id for spec in specs]}


def run_fresh_stage(config: dict, root: Path, directory: Path, *, max_new_jobs: int,
                    runner=run_training_pipeline) -> dict:
    """Execute at most three exact approved specs, serially, without failed-call retry.

    Read ``directory/stage.json``. Semantic proofs must carry run_id,
    experiment_id, verified and physical_interactions (actual environment calls,
    including warmup, never model processed samples). The transport runner owns
    reservations, call binding, cleanup and downloads; this driver owns no ledger.
    """
    if type(max_new_jobs) is not int or not 1 <= max_new_jobs <= 3:
        raise ValueError('Explicit max_new_jobs exact integer 1..3 required')
    root, directory = Path(root).resolve(), Path(directory).resolve()
    budget = [0, 0]
    descriptor = _strict_json(directory / 'stage.json', budget)
    fields = {'schema', 'stage_id', 'freeze_id', 'experiment_id', 'spec_paths',
              'max_new_jobs', 'approved', 'purpose', 'dependencies', 'output_root'}
    if (type(descriptor) is not dict or set(descriptor) != fields
            or descriptor['schema'] != 'fresh_benchmark_stage_v1'
            or descriptor['experiment_id'] != EXPERIMENT or descriptor['purpose'] != PURPOSE
            or descriptor['approved'] is not True
            or type(descriptor['max_new_jobs']) is not int
            or descriptor['max_new_jobs'] != max_new_jobs
            or type(descriptor['freeze_id']) is not str or not descriptor['freeze_id']):
        raise ValueError('Exact approved fresh stage descriptor required')
    safe_run_id(descriptor['stage_id'])
    output = _absolute(descriptor['output_root'])
    settings = config.get('native_acceptance', {}).get('fresh_benchmark', {})
    if (settings.get('experiment_id') != EXPERIMENT
            or not directory.is_relative_to(output) or directory == output):
        raise ValueError('Registered fresh experiment and exclusive stage output required')
    build_plan(config)
    paths = descriptor['spec_paths']
    if (type(paths) is not list or not paths or len(paths) > max_new_jobs
            or len(set(paths)) != len(paths)):
        raise ValueError('Nonempty unique bounded frozen specs required')
    specs = []
    for path in paths:
        absolute = _absolute(path)
        if not absolute.is_relative_to(output):
            raise ValueError('Specs must belong to fresh output root')
        specs.append(RunSpec(**_strict_json(absolute, budget)))
    if len({spec.run_id for spec in specs}) != len(specs):
        raise ValueError('Duplicate frozen run ID')
    for spec in specs:
        safe_run_id(spec.run_id)
        if (spec.config.get('experiment', {}).get('tag') != EXPERIMENT
                or spec.config.get('native_acceptance', {}).get('fresh_benchmark') != settings
                or spec.stage not in {'collect', 'benchmark_attack', 'audit', 'fit_risk',
                                      'train_policy', 'validate'}
                or spec.config.get('modal', {}).get('max_containers') != 1
                or type(spec.config['modal']['max_containers']) is not int
                or type(spec.config['modal'].get('retries')) is not int
                or spec.config['modal']['retries'] != 0):
            raise ValueError('Fresh registered scope and single-container no-retry specs required')
        if (spec.config.get('data', {}).get('split_seed') != 8501
                or type(spec.config['data']['split_seed']) is not int
                or spec.config['data'].get('partition_seed') != 8502
                or type(spec.config['data']['partition_seed']) is not int
                or spec.config['modal'].get('startup_timeout_seconds') != 180
                or type(spec.config['modal']['startup_timeout_seconds']) is not int
                or spec.config['modal'].get('volume_name') != settings.get('volume_name')
                or not settings.get('volume_name')):
            raise ValueError('Exact fresh split, partition, startup deadline and volume required')
    dependencies = descriptor['dependencies']
    if type(dependencies) is not list:
        raise ValueError('Explicit declared dependencies required')
    seen = set()
    for dependency in dependencies:
        if type(dependency) is not dict or set(dependency) != {'run_id', 'gate_path', 'gate_snapshot'}:
            raise ValueError('Exact dependency snapshot required')
        run_id = safe_run_id(dependency['run_id'])
        if run_id in seen or run_id in {spec.run_id for spec in specs}:
            raise ValueError('Duplicate or self dependency')
        seen.add(run_id)
        path = _absolute(dependency['gate_path'])
        if not path.is_relative_to(output):
            raise ValueError('Historical dependency forbidden')
        proof = _strict_json(path, budget)
        if proof != dependency['gate_snapshot']:
            raise ValueError('Dependency gate snapshot changed')
        _gate(proof, run_id)
        if not all(type(proof.get(key)) is str and proof[key] for key in ('call_id', 'app_id')):
            raise ValueError('Durable dependency call binding required')
        binding_path = _absolute(proof.get('call_binding_path'))
        if not binding_path.is_relative_to(output):
            raise ValueError('Historical dependency call binding forbidden')
        binding = _strict_json(binding_path, budget)
        if any(binding.get(key) != proof[key] for key in ('run_id', 'call_id', 'app_id')):
            raise ValueError('Dependency durable call binding mismatch')
    # This mandatory check verifies exact approved snapshots and CURRENT source.
    # Descriptor.approved is never sufficient authority to open an app.
    approvals = check_admission(specs, root)
    if any(witness.get('purpose') != PURPOSE or witness.get('freeze_id') != descriptor['freeze_id']
           for witness, *_ in approvals) or len(approvals) != len(specs):
        raise ValueError('Fresh stage admission binding mismatch')
    module, name = VERIFIER.split(':')
    if not callable(getattr(importlib.import_module(module), name, None)):
        raise ValueError('Fresh semantic verifier unavailable; execution blocked')
    with _controller_lock(directory):
        journal = directory / 'fresh-journal.json'
        identity = _journal_identity(descriptor, directory, specs)
        historical = False
        if journal.exists():
            old = _strict_json(journal, budget)
            historical = 'snapshot' in old
            matches = (old.get('snapshot') == {'descriptor': descriptor,
                       'specs': [asdict(spec) for spec in specs]} if historical else
                       all(old.get(key) == value for key, value in identity.items()))
            if not matches:
                raise ValueError('Fresh stage snapshot changed; never blindly retry')
        if any((directory / item).exists() for item in ('state.json', 'manifest.json', 'jobs')):
            raise ValueError('Historical controller phase recovery forbidden')
        state = {**identity, 'status': 'started', 'verification_refs': {}}
        if not historical:
            write_json(journal, state, replace=journal.exists())
        try:
            # The shared controller validates cached specs/receipts and admission.
            # It reattaches known calls; reserve() still protects unknown spawns.
            result = runner(specs, root, directory / 'transport', verify=VERIFIER,
                            artifact_destination=directory / 'artifacts', local_only=False)
            if result.get('status') != 'completed' or set(result.get('jobs', {})) != {s.run_id for s in specs}:
                raise ValueError('All fresh jobs must complete semantic verification')
            count = 0
            for spec in specs:
                row = result['jobs'][spec.run_id]
                if row.get('phase') != 'verified' or row.get('shutdown_verified') is not True:
                    raise ValueError('Verified semantic gate and exact cleanup required')
                count += _gate(row.get('verification'), spec.run_id)
                binding = _strict_json(directory / 'transport' / 'jobs' / spec.run_id / 'call.json', budget)
                if (binding.get('run_id') != spec.run_id or not row.get('call_id') or not row.get('app_id')
                        or binding.get('call_id') != row['call_id'] or binding.get('app_id') != row['app_id']):
                    raise ValueError('Durable current-call binding mismatch')
                # A JSON pointer references the authoritative controller proof.
                state['verification_refs'][spec.run_id] = (
                    str(directory / 'transport' / 'state.json') +
                    '#/jobs/' + spec.run_id + '/verification')
            state.update(status='completed', physical_interactions=count, actual_cloud_cost=None,
                         controller_ref=str(directory / 'transport' / 'state.json'))
            if not historical:
                write_json(journal, state, replace=True)
            return state
        except BaseException as error:
            state.update(status='blocked', error={'type': type(error).__name__, 'message': str(error)})
            if not historical:
                write_json(journal, state, replace=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--stage-directory', type=Path, required=True)
    parser.add_argument('--max-new-jobs', type=int, required=True)
    args = parser.parse_args()
    config = load_notebook_config(args.root / 'experiment.ipynb')
    run_fresh_stage(config, args.root, args.stage_directory, max_new_jobs=args.max_new_jobs)


if __name__ == '__main__':
    main()
