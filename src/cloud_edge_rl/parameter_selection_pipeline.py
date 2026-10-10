"""Independent, fail-closed DQN/PPO parameter-selection entry point.

Standalone config takes precedence over legacy notebook CONFIG. Plans and registered jobs are immutable snapshots,
not alternate configuration inputs. Existing training and transport are reused as-is.
"""

import argparse
import ast
import csv
import math
import sys
import uuid
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .config import RunSpec
from .parameter_selection import build_plan, candidate_config, execution_blockers, rank_results
from .transport import TransportLimits, download_artifacts, read_json, run_bounded, write_json


def _config_target(node):
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return isinstance(node, ast.Name) and node.id == 'CONFIG'


def load_notebook_config(path: Path) -> dict:
    """Load standalone CONFIG first; deprecated notebook fallback never runs jobs."""
    path = Path(path).resolve()
    root = path if path.is_dir() else path.parent
    standalone = root / 'config' / 'experiment.json'
    if path.name == 'experiment.json' or standalone.exists():
        from .config import load_config
        return load_config(path if path.name == 'experiment.json' else standalone)
    notebook = read_json(Path(path))
    statements, definitions = [], 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for statement in tree.body:
            assignment = isinstance(statement, ast.Assign) and all(
                _config_target(target) for target in statement.targets)
            update = (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                      and isinstance(statement.value.func, ast.Attribute)
                      and statement.value.func.attr == 'update'
                      and _config_target(statement.value.func.value))
            if not (assignment or update):
                continue
            if assignment:
                definitions += sum(isinstance(target, ast.Name) and target.id == 'CONFIG'
                                   for target in statement.targets)
            for node in ast.walk(statement):
                if isinstance(node, ast.Call):
                    function = node.func
                    allowed = (isinstance(function, ast.Name)
                               and function.id in {'dict', 'list', 'tuple', 'zip'})
                    allowed |= (isinstance(function, ast.Attribute)
                                and function.attr in {'items', 'update'}
                                and _config_target(function.value))
                    if not allowed:
                        raise ValueError('Only literal CONFIG construction and updates are allowed')
                if isinstance(node, ast.Attribute) and node.attr not in {'items', 'update'}:
                    raise ValueError('Unexpected attribute in CONFIG')
            statements.append(statement)
    if definitions != 1:
        raise ValueError('Exactly one notebook CONFIG definition required')
    namespace = {'__builtins__': {'dict': dict, 'list': list, 'tuple': tuple, 'zip': zip}}
    code = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(compile(code, str(path), 'exec'), namespace)
    if type(namespace['CONFIG']) is not dict:
        raise ValueError('Notebook CONFIG must be a dictionary')
    return namespace['CONFIG']


def prepare_selection(config: dict, root: Path, directory: Path | None = None) -> dict:
    """Write twenty design snapshots with no model loads, SDK calls or approvals."""
    settings = config['native_acceptance']['parameter_selection']
    directory = (Path(directory) if directory is not None
                 else Path(root) / settings['output_directory']).resolve()
    plan = build_plan(config)
    manifest = {'plan': plan, 'config_snapshot': deepcopy(config),
                'execution_blockers': execution_blockers(config),
                'planned_reference_interactions': 60, 'planned_max_compute_calls': 62,
                'new_actual_interactions': 0, 'new_compute_calls': 0,
                'new_cloud_cost_usd': 0.0, 'historical_actual_cloud_cost_usd': None,
                'status': 'design_registered_not_executed'}
    path = directory / 'registration.json'
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError('Parameter-selection snapshot changed; use a new registration directory')
        for row in plan['candidates']:
            if read_json(directory / 'candidates' / f"{row['id']}.json") != candidate_config(config, row['id']):
                raise ValueError('Registered candidate snapshot changed')
    else:
        directory.mkdir(parents=True, exist_ok=True)
        for row in plan['candidates']:
            candidate_path = directory / 'candidates' / f"{row['id']}.json"
            value = candidate_config(config, row['id'])
            if candidate_path.exists():
                if read_json(candidate_path) != value:
                    raise ValueError('Partial candidate registration changed; never overwrite it')
            else:
                write_json(candidate_path, value)
        write_json(path, manifest)
    state_path = directory / 'state.json'
    if state_path.exists():
        state = read_json(state_path)
        if (type(state) is not dict or type(state.get('status')) is not str
                or type(state.get('jobs')) is not dict or state.get('automatic_retry') is not False):
            raise ValueError('Invalid retained parameter-selection state')
    else:
        if any((directory / stage).exists() for stage in ('training', 'validation')):
            raise ValueError('Missing state after possible execution; consumption cannot be reset to zero')
        write_json(state_path, {'status': 'not_executed', 'jobs': {},
                                'actual_interactions': 0, 'automatic_retry': False})
    return manifest


def prepare_parameter_specs(config: dict, root: Path) -> list[RunSpec]:
    """Return the exact registered graph for Experiment.prepare, without execution.

    Retains the historical scientific-design and dependency checks; admission and
    semantic verification remain owned by the common controller at submission.
    """
    trains, evaluations, references = _registered_jobs(config, Path(root).resolve())
    return [*trains.values(), *evaluations.values(), *references.values()]


def _scientific(config):
    config = deepcopy(config)
    config.get('native_acceptance', {}).pop('execution_admission', None)
    config.get('experiment', {}).pop('variant', None)
    config['execution'].pop('split', None)
    return config


def _registered_jobs(config, root):
    settings = config['native_acceptance']['parameter_selection']
    paths = settings.get('registered_spec_paths')
    if type(paths) is not list or len(paths) != 62 or any(type(path) is not str for path in paths):
        raise ValueError('Register exact 20 training, 40 validation and 2 reference specs first')
    specs = [RunSpec(**read_json(Path(root) / path)) for path in paths]
    descriptors = settings.get('shared_execution_descriptors')
    if descriptors is not None:
        from .comparisons.acceptance import _first_artifact
        if type(descriptors) is not dict or set(descriptors) != {'source', 'admission'}:
            raise ValueError('Exactly two frozen shared execution descriptor paths required')
        source_descriptor = read_json(Path(root) / descriptors['source'])
        admission_descriptor = read_json(Path(root) / descriptors['admission'])
        source = _first_artifact(source_descriptor)
        _first_artifact(admission_descriptor)
        for spec in specs:
            if (spec.provenance.get('identity_kind') != 'source_contents_no_hash'
                    or spec.provenance.get('source_contents_path') != source_descriptor['path']
                    or 'source_contents' in spec.provenance
                    or 'execution_admission' in spec.config['native_acceptance']):
                raise ValueError('Compact spec differs from its explicit shared source descriptor')
            spec.provenance['source_contents'] = source
            spec.config['native_acceptance']['execution_admission'] = admission_descriptor
    if len({spec.run_id for spec in specs}) != 62:
        raise ValueError('Registered run IDs must be unique')
    trains, evaluations, references = {}, {}, {}
    plan = build_plan(config)
    algorithms = {row['id']: row['algorithm'] for row in plan['candidates']}
    for spec in specs:
        cid = spec.config.get('experiment', {}).get('variant')
        if (spec.config['execution']['backend'] != 'modal'
                or spec.config['risk'].get('mode') != 'fitted'
                or spec.risk_version != settings['frozen_risk_version']
                or spec.inputs.get('risk_checkpoint') != settings['frozen_risk_checkpoint']
                or spec.config['rl'].get('gamma') != 1
                or spec.config['modal'].get('max_containers') != 1):
            raise ValueError('Frozen risk, finite-horizon and single-container settings required')
        if spec.stage == 'train_policy':
            if (cid not in algorithms or cid in trains or spec.method != algorithms[cid]
                    or spec.seed != settings['training_seed']
                    or spec.config['execution']['split'] != 'train'
                    or spec.inputs.get('previous_training_checkpoint') not in (None, '')
                    or _scientific(spec.config) != _scientific(candidate_config(config, cid))):
                raise ValueError('Training spec differs from registered parameter design')
            trains[cid] = spec
        elif spec.stage == 'validate' and spec.config['execution']['split'] == 'validation':
            if spec.seed not in settings['evaluation_seeds']:
                raise ValueError('Evaluation seed is not preregistered')
            if spec.method == 'time_greedy' and cid == 'reference-time_greedy':
                if spec.seed in references:
                    raise ValueError('Duplicate reference trajectory')
                if _scientific(spec.config) != _scientific(candidate_config(config, 'D00')):
                    raise ValueError('Reference scientific conditions changed')
                references[spec.seed] = spec
            else:
                key = (cid, spec.seed)
                if (cid not in algorithms or key in evaluations or spec.method != algorithms[cid]
                        or _scientific(spec.config) != _scientific(candidate_config(config, cid))):
                    raise ValueError('Evaluation differs from its frozen training configuration')
                evaluations[key] = spec
        else:
            raise ValueError('Parameter selection forbids sealed-test or unrelated stages')
    if (set(trains) != set(algorithms) or len(evaluations) != 40
            or set(references) != set(settings['evaluation_seeds'])):
        raise ValueError('Incomplete registered training/evaluation matrix')
    for (cid, _), spec in evaluations.items():
        if spec.inputs.get('agent_checkpoint') != f'runs/{trains[cid].run_id}/agent.pt':
            raise ValueError('Evaluation must load its own exact final 500-step checkpoint')
    return trains, evaluations, references


def verify_validation_artifacts(spec: RunSpec, artifacts: Path, result: dict) -> dict:
    if (spec.stage != 'validate' or spec.config['execution']['split'] != 'validation'
            or read_json(artifacts / 'config.json') != asdict(spec)
            or read_json(artifacts / 'status.json').get('state') != 'completed'):
        raise ValueError('Exact completed validation config/source required')
    data = read_json(artifacts / 'data_metadata.json')
    split = data['split']
    train, validation = split['train_source_indices'], split['validation_source_indices']
    clients = split['client_indices']
    if (data.get('official_test_included') is not False or data['rows']['official_test'] != 0
            or split['config'] != spec.config['data'] or set(train) & set(validation)
            or len(set(train)) != len(train) or len(set(validation)) != len(validation)
            or len(train) != data['rows']['train'] or len(validation) != data['rows']['validation']
            or sum(map(len, clients)) != len(train)
            or set().union(*map(set, clients)) != set(range(len(train)))):
        raise ValueError('Validation/train/private-client isolation mismatch')
    runtime = read_json(artifacts / 'runtime.json')
    gpu = spec.config['modal']['gpu']
    if gpu and (runtime.get('cuda_available') is not True or gpu not in runtime.get('gpu', '')):
        raise ValueError('Requested evaluation GPU not observed')
    with (artifacts / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    summary = read_json(artifacts / 'summary.json')
    rounds = len(rows)
    flags = ('is_warmup', 'terminated', 'truncated')
    if any(row[key].lower() not in ('true', 'false') for row in rows for key in flags):
        raise ValueError('Invalid validation task/warmup flags')
    terminal = [row['terminated'].lower() == 'true' or row['truncated'].lower() == 'true' for row in rows]
    if (not 1 <= rounds <= spec.config['environment']['max_rounds']
            or [int(row['round']) for row in rows] != list(range(1, rounds + 1))
            or len({row['episode'] for row in rows}) != 1
            or [row['is_warmup'].lower() == 'true' for row in rows] != [step <= 12 for step in range(1, rounds + 1)]
            or not terminal[-1] or any(terminal[:-1])
            or summary.get('run_id') != spec.run_id or summary.get('round') != rounds
            or result.get('run_id') != spec.run_id or summary.get('risk_version') != spec.risk_version):
        raise ValueError('Exactly one naturally completed validation trajectory required')
    for row in rows:
        if row['risk_version'] != spec.risk_version or any(not math.isfinite(float(row[key]))
                for key in ('elapsed_seconds', 'energy_joules', 'validation_accuracy', 'validation_loss')):
            raise ValueError('Invalid validation metrics or risk version')
    for key in ('elapsed_seconds', 'energy_joules', 'validation_accuracy'):
        value = summary[key]
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or value != float(rows[-1][key]) or result.get(key) != value):
            raise ValueError('Validation summary/metrics/compute return disagree')
    if summary['validation_accuracy'] > 1:
        raise ValueError('Invalid validation accuracy')
    return {'verified': True, 'run_id': spec.run_id, 'seed': spec.seed,
            'accuracy': summary['validation_accuracy'], 'time_seconds': summary['elapsed_seconds'],
            'energy_joules': summary['energy_joules'], 'actual_interactions': rounds,
            'warmup_steps': min(12, rounds), 'task_terminal': True, 'official_test_included': False,
            'predicted_exposure_risk': summary.get('predicted_exposure_risk'),
            'complete_true_risk': None, 'formal_acceptance': 'not_evaluated'}


def verify_parameter_training(spec: RunSpec, artifacts: Path, result: dict) -> dict:
    """Extend, not change, the native gate with this batch's actual learning clocks."""
    import torch

    from .training_pipeline import verify_training_artifacts

    proof = verify_training_artifacts(spec, artifacts, result)
    state = torch.load(artifacts / 'training.pt', map_location='cpu', weights_only=False)['state']
    with (artifacts / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    learned = state['learned_steps']
    environment = state['environment']
    if (len(rows) != 500 or state['environment_steps'] != 500
            or state['warmup_steps'] + learned != 500
            or sum(row['is_warmup'].lower() == 'true' for row in rows) != state['warmup_steps']
            or any(row[key].lower() not in ('true', 'false') for row in rows
                   for key in ('is_warmup', 'terminated', 'truncated'))
            or type(state['finished']) is not bool or environment['finished'] is not state['finished']
            or (rows[-1]['terminated'].lower() == 'true' or rows[-1]['truncated'].lower() == 'true') != state['finished']
            or int(rows[-1]['episode']) != state['episode']
            or int(rows[-1]['round']) != environment['round_index']):
        raise ValueError('Actual 500-interaction task and warmup counters disagree')
    rl = spec.config['rl']
    if spec.method == 'dqn':
        expected_updates = sum(step >= max(rl['learning_starts'], rl['batch_size'])
                               and step % rl['train_frequency'] == 0
                               and rl['buffer_size'] >= rl['batch_size']
                                for step in range(1, learned + 1))
    else:
        final_step_is_warmup = rows[-1]['is_warmup'].lower() == 'true'
        expected_updates = (learned // rl['rollout_steps'] if final_step_is_warmup
                            else math.ceil(learned / rl['rollout_steps']))
        expected_rollout = learned % rl['rollout_steps'] if final_step_is_warmup else 0
        if len(state['rollout']) != expected_rollout:
            raise ValueError('Existing PPO rollout/flush clock was changed')
    if expected_updates < 1 or state['updates'] != expected_updates:
        raise ValueError('Native optimizer update clock differs from registered short budget')
    return {**proof, 'warmup_steps': state['warmup_steps'], 'learned_steps': state['learned_steps'],
            'optimizer_updates': state['updates'], 'final_task_rounds': environment['round_index'],
            'final_task_terminal': state['finished'], 'budget_does_not_force_terminal': True}


def _evaluate(spec, root, directory, destination, limits, worker=run_bounded,
              downloader=download_artifacts):
    from .training_pipeline import _compute_receipt

    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (('spec.json', asdict(spec)), ('transport.json', asdict(limits))):
        path = directory / name
        if path.exists():
            if read_json(path) != value:
                raise ValueError('Frozen evaluation job changed on resume')
        else:
            write_json(path, value)
    common = [sys.executable, '-m', 'cloud_edge_rl.parameter_selection_pipeline',
              '--root', str(root), '--stage', str(directory)]
    try:
        receipt = _compute_receipt(directory, spec)
        if receipt is None:
            try:
                worker([*common, '--action', 'compute'], spec.config['modal']['timeout_seconds']
                       + spec.config['modal']['startup_timeout_seconds'] + limits.cleanup_seconds,
                       directory / f'compute-{uuid.uuid4().hex}.log')
            except Exception:
                if _compute_receipt(directory, spec) is None:
                    raise
            receipt = _compute_receipt(directory, spec)
            if receipt is None:
                raise ValueError('Evaluation compute receipt missing')
        shutdown = directory / 'shutdown.json'
        if not shutdown.exists():
            worker([*common, '--action', 'cleanup'], limits.cleanup_seconds + 10,
                   directory / f'cleanup-{uuid.uuid4().hex}.log')
        proof = read_json(shutdown)
        if proof.get('app_id') != receipt['app_id'] or proof.get('verified') is not True or proof.get('stopped') is not True:
            raise ValueError('Evaluation app shutdown not verified')
        artifacts = downloader(spec.config['modal']['volume_name'], spec.run_id, destination, limits=limits)
        output = directory / f'verification-{uuid.uuid4().hex}.json'
        worker([*common, '--action', 'verify-validation', '--artifacts', str(artifacts),
                '--verification-output', str(output)], limits.verification_seconds,
               directory / f'verify-{uuid.uuid4().hex}.log')
        proof = read_json(output)
        if proof.get('verified') is not True or proof.get('run_id') != spec.run_id:
            raise ValueError('Evaluation semantic gate failed')
        return proof
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error), 'automatic_retry': False}
        if not (directory / 'shutdown.json').exists() and any(
                (directory / name).exists() for name in ('app.json', 'call.json')):
            try:
                worker([*common, '--action', 'cleanup'], limits.cleanup_seconds + 10,
                       directory / f'failure-cleanup-{uuid.uuid4().hex}.log')
            except Exception as cleanup_error:
                failure['cleanup_error'] = {'type': type(cleanup_error).__name__,
                                            'message': str(cleanup_error)}
        write_json(directory / f'failure-{uuid.uuid4().hex}.json', failure)
        raise


def run_parameter_selection(config: dict, root: Path, directory: Path, *,
                            training_runner=None, evaluation_runner=None,
                            max_new_jobs: int | None = None) -> dict:
    """All twenty jobs are frozen/preflighted first; no implicit approval or retries."""
    blockers = execution_blockers(config)
    if blockers:
        raise ValueError('Parameter selection blocked: ' + '; '.join(blockers))
    registered_bound = config['native_acceptance']['parameter_selection'].get('stage_execution', {}).get('max_new_jobs')
    if (type(registered_bound) is not int or registered_bound < 1
            or type(max_new_jobs) is not int or not 1 <= max_new_jobs <= registered_bound):
        raise ValueError('Stage job bound must be an explicit positive integer within the registered bound')
    from .admission import check_admission
    from .training_pipeline import _controller_lock, run_training_pipeline

    root, directory = Path(root).resolve(), Path(directory).resolve()
    registration = read_json(directory / 'registration.json')
    if registration['config_snapshot'] != config:
        raise ValueError('Sole CONFIG differs from the registered selection snapshot')
    trains, evaluations, references = _registered_jobs(config, root)
    # Existing shared paid guards are deliberately not bypassed or relaxed here.
    check_admission([*trains.values(), *evaluations.values(), *references.values()], root)
    training_runner = run_training_pipeline if training_runner is None else training_runner
    evaluation_runner = _evaluate if evaluation_runner is None else evaluation_runner
    limits = TransportLimits(**config['modal']['transport'])
    artifacts = directory / 'artifacts'
    state_path = directory / 'state.json'
    with _controller_lock(directory):
        state = read_json(state_path)
        state.update(status='running', actual_interactions=None, observed_completed_interactions=0)
        try:
            outcomes = {}
            observed_interactions = 0
            new_jobs = 0

            def stage_complete(phase):
                state.update(status='stage_completed', active_phase=phase,
                             automatic_retry=False, actual_interactions=None,
                             observed_completed_interactions=observed_interactions)
                write_json(state_path, state, replace=True)
                return {'status': 'stage_completed', 'phase': phase,
                        'new_jobs_completed': new_jobs,
                        'observed_completed_interactions': observed_interactions,
                        'formal_acceptance': 'not_evaluated', 'official_test_access': False}

            for cid, spec in trains.items():
                prior = state['jobs'].get(cid, {}).get('training_verified') is True
                if not prior and max_new_jobs is not None and new_jobs >= max_new_jobs:
                    return stage_complete('training')
                state.update(active_candidate=cid, active_run_id=spec.run_id, active_phase='training')
                write_json(state_path, state, replace=True)
                trained = training_runner([spec], root, directory / 'training' / cid,
                                          artifact_destination=artifacts, transport_limits=limits,
                                          verify='cloud_edge_rl.parameter_selection_pipeline:verify_parameter_training')
                proof = trained['jobs'][spec.run_id]['verification']
                if proof.get('verified') is not True or proof.get('actual_interactions') != 500:
                    raise ValueError('Final exact 500-interaction training gate required')
                outcomes[cid] = {'configuration_id': cid, 'algorithm': spec.method,
                                 'status': 'completed', 'artifact_verified': True,
                                 'actual_interactions': proof['actual_interactions'],
                                 'warmup_steps': proof['warmup_steps'], 'learned_steps': proof['learned_steps'],
                                 'optimizer_updates': proof['optimizer_updates'], 'evaluations': []}
                state['jobs'][cid] = {'training_run_id': spec.run_id, 'training_verified': True}
                new_jobs += not prior
                observed_interactions += proof['actual_interactions']
                state['observed_completed_interactions'] = observed_interactions
                write_json(state_path, state, replace=True)
            reference = {'method': 'time_greedy', 'artifact_verified': True, 'evaluations': []}
            validated = state.setdefault('validation_jobs', {})
            for spec in references.values():
                prior = validated.get(spec.run_id) is True
                if not prior and max_new_jobs is not None and new_jobs >= max_new_jobs:
                    return stage_complete('reference_validation')
                state.update(active_run_id=spec.run_id, active_phase='reference_validation')
                write_json(state_path, state, replace=True)
                evidence = evaluation_runner(spec, root, directory / 'validation' / spec.run_id, artifacts, limits)
                reference['evaluations'].append(evidence)
                validated[spec.run_id] = True
                new_jobs += not prior
                observed_interactions += evidence['actual_interactions']
                state['observed_completed_interactions'] = observed_interactions
                write_json(state_path, state, replace=True)
            for (cid, _), spec in evaluations.items():
                prior = validated.get(spec.run_id) is True
                if not prior and max_new_jobs is not None and new_jobs >= max_new_jobs:
                    return stage_complete('candidate_validation')
                state.update(active_run_id=spec.run_id, active_phase='candidate_validation')
                write_json(state_path, state, replace=True)
                evidence = evaluation_runner(spec, root, directory / 'validation' / spec.run_id, artifacts, limits)
                outcomes[cid]['evaluations'].append(evidence)
                validated[spec.run_id] = True
                new_jobs += not prior
                observed_interactions += evidence['actual_interactions']
                state['observed_completed_interactions'] = observed_interactions
                write_json(state_path, state, replace=True)
            result = rank_results(registration['plan'], list(outcomes.values()), reference)
            write_json(directory / f'results-{uuid.uuid4().hex}.json', result)
            state.update(status=result['status'], formal_acceptance='not_evaluated',
                         actual_interactions=observed_interactions)
            write_json(state_path, state, replace=True)
            return result
        except BaseException as error:
            state.update(status='blocked', automatic_retry=False,
                         error={'type': type(error).__name__, 'message': str(error)})
            write_json(state_path, state, replace=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'check', 'run', 'report', 'compute',
                                          'cleanup', 'verify-validation'), required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--verification-output', type=Path)
    parser.add_argument('--outcomes', type=Path)
    parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action in {'compute', 'cleanup', 'verify-validation'}:
        if args.stage is None:
            parser.error('Worker action requires --stage')
        from .training_pipeline import _cleanup_binding, _compute_receipt, execute_compute
        from .transport import stop_and_verify
        spec = RunSpec(**read_json(args.stage / 'spec.json'))
        if args.action == 'compute':
            execute_compute(spec, root, args.stage)
        elif args.action == 'cleanup':
            binding = _cleanup_binding(spec, root, args.stage)
            limits = TransportLimits(**read_json(args.stage / 'transport.json'))
            write_json(args.stage / 'shutdown.json', stop_and_verify(binding['app_id'], limits=limits))
        else:
            if args.artifacts is None or args.verification_output is None:
                parser.error('Verification requires artifacts and an exclusive output')
            receipt = _compute_receipt(args.stage, spec)
            if receipt is None:
                raise ValueError('Completed evaluation receipt required')
            write_json(args.verification_output,
                       verify_validation_artifacts(spec, args.artifacts, receipt['result']))
        return
    config = load_notebook_config(root / 'experiment.ipynb')
    directory = root / config['native_acceptance']['parameter_selection']['output_directory']
    if args.action == 'prepare':
        result = prepare_selection(config, root, directory)
    elif args.action == 'check':
        result = {'status': 'blocked' if execution_blockers(config) else 'locally_ready',
                  'execution_blockers': execution_blockers(config), 'no_compute_submitted': True}
    elif args.action == 'run':
        result = run_parameter_selection(config, root, directory, max_new_jobs=args.max_new_jobs)
    else:
        if args.outcomes is None:
            parser.error('report requires --outcomes, never selects from missing runs')
        evidence = read_json(args.outcomes)
        result = rank_results(build_plan(config), evidence['rows'], evidence.get('reference'))
    import json
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
