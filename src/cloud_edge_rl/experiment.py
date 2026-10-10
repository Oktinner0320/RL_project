"""The single notebook-facing experiment lifecycle.

Planning is inert. Submission retains the existing admission and reservation
barriers. Cloud computation, artifact transfer and local closure are separate
durable states: a failed download can never start another training call.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from .config import RunSpec, source_identity
from .experiment_graph import (MAX_METADATA, expand, freeze_graph, run_graph,
                               validate_graph)
from .transport import read_json, run_bounded, write_json


class Experiment:
    def __init__(self, directory, root):
        self.directory, self.root = Path(directory).resolve(), Path(root).resolve()
        path = self.directory / 'snapshot.json'
        if path.stat().st_size > MAX_METADATA:
            raise ValueError('Frozen graph exceeds metadata bound')
        self.graph = read_json(path)
        validate_graph(self.graph)

    @classmethod
    def open(cls, directory, root=None):
        return cls(directory, Path.cwd() if root is None else root)

    @classmethod
    def prepare(cls, config, root, specs=None, verifiers=None, directory=None):
        root = Path(root).resolve()
        controls = config['workflow']['experiment']
        target = Path(directory) if directory is not None else root / controls['directory']
        target = target.resolve()
        if (target / 'snapshot.json').exists():
            saved = cls(target, root)
            if read_json(target / 'configuration.json') != config:
                raise ValueError('Frozen configuration changed; reopen the original experiment')
            if specs is not None:
                from dataclasses import asdict
                actual = [expand(saved.graph['pool'], n['spec_ref']) for n in saved.graph['nodes']]
                if actual != [asdict(s) for s in specs]:
                    raise ValueError('Frozen run specifications changed')
            return saved
        if (target / 'submission.json').exists():
            raise ValueError('Missing immutable snapshot after possible execution')
        dependencies = None
        if specs is None:
            mode = controls['mode']
            if mode == 'fresh':
                from .fresh_benchmark import build_plan, build_run_specs
                plan = build_plan(config)
                identities = {n['id']: f"{n['phase']}-{uuid.uuid4().hex}" for n in plan['nodes']}
                specs = build_run_specs(config, root, identities, source_identity(root))
                dependencies = {identities[n['id']]: [identities[d] for d in n['dependencies']]
                                for n in plan['nodes']}
                verifiers = verifiers or 'cloud_edge_rl.fresh_benchmark:verify_fresh_artifacts'
            elif mode == 'selection':
                from .parameter_selection_pipeline import prepare_parameter_specs
                specs = prepare_parameter_specs(config, root)
            elif mode == 'comparisons':
                from .comparisons.workflow import prepare_comparison_specs
                specs = prepare_comparison_specs(config, root)
            else:
                raise ValueError('Unknown experiment planning mode')
        if len(specs) > controls.get('max_nodes', 256):
            raise ValueError('Configured graph node limit exceeded')
        if verifiers is None:
            raise ValueError('Explicit per-stage semantic verifiers required')
        graph = freeze_graph(specs, experiment_id=f'experiment-{uuid.uuid4().hex}',
                             verifier=verifiers, node_dependencies=dependencies,
                             node_seconds=controls['node_timeout_seconds'],
                             verify_seconds=controls['verification_timeout_seconds'],
                             graph_seconds=controls['graph_timeout_seconds'])
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / 'configuration.json'
        if config_path.exists():
            if read_json(config_path) != config:
                raise ValueError('Partial preparation configuration changed')
        else:
            write_json(config_path, config)
        write_json(target / 'snapshot.json', graph)
        return cls(target, root)

    def _controls(self):
        return read_json(self.directory / 'configuration.json')['workflow']['experiment']

    def _command(self, action, timeout):
        return run_bounded([sys.executable, '-m', 'cloud_edge_rl.experiment',
                            '--action', action, '--directory', str(self.directory),
                            '--root', str(self.root)], timeout,
                           self.directory / 'controller' / f'{action}-{uuid.uuid4().hex}.log')

    def submit(self):
        controls = self._controls()
        if controls['submission_enabled'] is not True:
            raise ValueError('Submission is disabled in the frozen configuration')
        receipt = self.directory / 'submission.json'
        if receipt.exists():
            saved = read_json(receipt)
            if not saved.get('call_id'):
                raise ValueError('Submission identity unknown; explicit reconciliation required')
            return self.status()
        self._command('submit', controls['controller_timeout_seconds'])
        return self.status()

    def resume(self):
        """Explicit continuation only after the original call returned safely."""
        if self._controls()['submission_enabled'] is not True:
            raise ValueError('Submission disabled')
        self._command('resume', self._controls()['controller_timeout_seconds'])
        return self.status()

    def cancel(self):
        """Stop only this experiment's recorded call/app; retain unknown consumption."""
        self._command('cancel', 120)
        return self.status()

    def status(self, refresh=False):
        if refresh:
            self._command('status', 60)
        result = {'experiment_id': self.graph['experiment_id'],
                  'planned_nodes': len(self.graph['nodes']), 'status': 'prepared',
                  'formal_acceptance': 'not_evaluated', 'actual_cloud_cost_usd': None}
        for name in ('submission', 'remote-status', 'transfer', 'closure', 'cancellation'):
            path = self.directory / f'{name}.json'
            if path.exists():
                value = read_json(path)
                if name == 'submission' or value.get('attempt_id') == result.get('submission', {}).get('attempt_id'):
                    result[name] = value
        if 'cancellation' in result:
            result['status'] = 'cancelled_consumption_requires_reconciliation'
        elif 'closure' in result:
            result['status'] = result['closure']['status']
        elif 'remote-status' in result:
            result['status'] = result['remote-status']['status']
        elif 'submission' in result:
            result['status'] = result['submission']['status']
        return result

    def collect(self):
        from .experiment_transfer import collect_experiment
        submission = read_json(self.directory / 'submission.json')
        self.status(refresh=True)
        remote = read_json(self.directory / 'remote-status.json')
        if remote['status'] == 'running':
            raise ValueError('Cloud graph still running; terminal inventory required')
        if not remote.get('call_completed'):
            raise ValueError('Cloud completion unknown; reconcile original call before transfer')
        # Exact app cleanup precedes the one logical artifact return stage.
        shutdown = self.directory / f"shutdown-{submission['attempt_id']}.json"
        if not shutdown.exists():
            self._command('cleanup', 120)
        stopped = read_json(shutdown)
        if stopped.get('stopped') is not True or stopped.get('app_id') != submission['app_id']:
            raise ValueError('Exact app shutdown not verified')
        destination = self.directory / 'artifacts' / submission['attempt_id']
        result = collect_experiment(submission['volume_name'], self.graph['experiment_id'],
                                    destination, environment=submission['environment'])
        result['attempt_id'] = submission['attempt_id']
        write_json(self.directory / 'transfer.json', result, replace=True)
        return result

    def verify(self):
        """Entirely local closure; never opens Modal or starts compute."""
        transfer = read_json(self.directory / 'transfer.json')
        submission = read_json(self.directory / 'submission.json')
        if (transfer.get('transport_complete') is not True
                or transfer.get('experiment_id') != self.graph['experiment_id']
                or transfer.get('attempt_id') != submission['attempt_id']):
            raise ValueError('Unified artifact transfer incomplete or stale')
        artifacts = self.directory / 'artifacts' / submission['attempt_id']
        if Path(transfer['destination']).resolve() != artifacts.resolve():
            raise ValueError('Artifact transfer destination mismatch')
        base = artifacts / 'experiments' / self.graph['experiment_id']
        if read_json(base / 'snapshot.json') != self.graph:
            raise ValueError('Returned graph differs from the frozen plan')
        state = read_json(base / 'state.json')
        if state['status'] != 'cloud_verified':
            raise ValueError('Full experiment has not reached cloud verification')
        closure = self.directory / 'closure-progress.json'
        proofs = []
        for node in self.graph['nodes']:
            if state['nodes'][node['run_id']]['status'] != 'verified':
                raise ValueError('Unverified cloud node')
            # Reverify all scientific evidence against the returned dependency tree.
            run_bounded([sys.executable, '-m', 'cloud_edge_rl.experiment_graph',
                         '--snapshot', str(self.directory / 'snapshot.json'),
                         '--workspace', str(artifacts), '--run-id', node['run_id'],
                         '--phase', 'verify'], node['verify_seconds'],
                        self.directory / 'closure-logs' / f"{node['run_id']}-{uuid.uuid4().hex}.log")
            proofs.append({'run_id': node['run_id'], 'verified': True})
            write_json(closure, {'attempt_id': submission['attempt_id'], 'nodes': proofs}, replace=True)
        result = {'experiment_id': self.graph['experiment_id'], 'status': 'completed',
                  'attempt_id': submission['attempt_id'],
                  'nodes': proofs, 'formal_acceptance': 'not_evaluated',
                  'actual_cloud_cost_usd': None}
        write_json(self.directory / 'closure.json', result, replace=True)
        return result


def run_remote_graph(graph, volume_name, attempt_id):
    """Detached call: commit progress independently of notebook connectivity."""
    import modal
    volume = modal.Volume.from_name(volume_name)
    volume.reload()
    from .experiment_graph import publish_inventory
    from .transport import safe_run_id
    safe_run_id(attempt_id)
    workspace = Path('/workspace')
    result = run_graph(graph, workspace, commit=volume.commit)
    write_json(workspace / 'experiments' / graph['experiment_id'] / 'calls' / f'{attempt_id}.json',
               {'attempt_id': attempt_id, 'state': result})
    publish_inventory(graph, workspace)
    volume.commit()
    return result


def _remote_state(experiment, submission):
    """Use a durable attempt receipt when the SDK's transient result expired."""
    import modal
    try:
        state = modal.FunctionCall.from_id(submission['call_id']).get(timeout=0)
    except TimeoutError:
        raise
    except Exception as original:
        from .transport import _decode_json
        volume = modal.Volume.from_name(submission['volume_name'],
                                        environment_name=submission['environment'])
        content = bytearray()
        path = f"experiments/{experiment.graph['experiment_id']}/calls/{submission['attempt_id']}.json"
        try:
            for chunk in volume.read_file(path):
                if len(content) + len(chunk) > 2 * 1024**2:
                    raise ValueError('Bounded remote completion receipt exceeded')
                content.extend(chunk)
            receipt = _decode_json(content.decode('utf-8'))
            if receipt['attempt_id'] != submission['attempt_id']:
                raise ValueError('Remote completion attempt mismatch')
            state = receipt['state']
        except Exception as error:
            raise RuntimeError('Original call and durable completion unavailable; no resubmission') from original
    if (type(state) is not dict or state.get('experiment_id') != experiment.graph['experiment_id']
            or set(state.get('nodes', {})) != {node['run_id'] for node in experiment.graph['nodes']}
            or state.get('status') not in ('cloud_verified', 'paused_deadline', 'blocked')):
        raise ValueError('Remote experiment completion identity/state mismatch')
    return state


def _submit(experiment, *, continuation=False):
    from .admission import _locked
    with _locked(experiment.directory / 'submission.json'):
        return _submit_locked(experiment, continuation=continuation)


def _submit_locked(experiment, *, continuation=False):
    import modal
    from .experiment_admission import validate_and_reserve_graph
    from .modal_runner import create_app

    graph, directory = experiment.graph, experiment.directory
    if experiment._controls()['submission_enabled'] is not True:
        raise ValueError('Submission disabled in the frozen configuration')
    submission = directory / 'submission.json'
    specs = [RunSpec(**expand(graph['pool'], node['spec_ref'])) for node in graph['nodes']]
    if continuation:
        prior = read_json(submission)
        result = _remote_state(experiment, prior)
        if result.get('status') not in ('paused_deadline', 'blocked'):
            raise ValueError('Only an observed paused or verification-blocked graph can continue')
        if any(row['status'] not in ('pending', 'verified', 'computed', 'verification_failed')
               for row in result['nodes'].values()):
            raise ValueError('Ambiguous compute cannot be automatically retried')
        from .transport import stop_and_verify
        stopped = stop_and_verify(prior['app_id'], environment=prior['environment'])
        if stopped.get('stopped') is not True or stopped.get('app_id') != prior['app_id']:
            raise ValueError('Prior graph app shutdown not verified')
        write_json(directory / f"prior-shutdown-{uuid.uuid4().hex}.json", stopped)
        write_json(directory / f"prior-submission-{uuid.uuid4().hex}.json", prior)
    elif submission.exists():
        raise ValueError('Existing submission cannot be repeated')
    # Preserve all historical approvals, allocations, failed attempts and source checks.
    validate_and_reserve_graph(graph, experiment.root, directory)
    settings = specs[0].config['modal']
    record = {'experiment_id': graph['experiment_id'], 'status': 'reserved',
              'attempt_id': uuid.uuid4().hex,
              'call_id': None, 'app_id': None, 'volume_name': settings['volume_name'],
              'environment': settings.get('environment', 'main'),
              'reserved_run_ids': [s.run_id for s in specs],
              'actual_cloud_cost_usd': None, 'historical_consumption_reset': False}
    write_json(submission, record, replace=continuation)
    app, function, _ = create_app(experiment.root, settings, specs,
                                  graph_timeout_seconds=graph['graph_seconds'],
                                  graph_controller_directory=directory)
    with modal.enable_output(), app.run(detach=True):
        record.update(status='submitting', app_id=app.app_id)
        write_json(submission, record, replace=True)
        call = function.spawn(graph, settings['volume_name'], record['attempt_id'])
        record.update(status='submitted', call_id=call.object_id)
        write_json(submission, record, replace=True)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('submit', 'resume', 'status', 'cleanup', 'cancel'), required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    experiment = Experiment.open(args.directory, args.root)
    if args.action in ('submit', 'resume'):
        _submit(experiment, continuation=args.action == 'resume')
        return
    submission = read_json(args.directory / 'submission.json')
    if not submission.get('call_id') or not submission.get('app_id'):
        raise ValueError('Submission identity unknown')
    if args.action in ('cleanup', 'cancel'):
        if args.action == 'cancel':
            import modal
            modal.FunctionCall.from_id(submission['call_id']).cancel(terminate_containers=True)
        from .transport import stop_and_verify
        result = stop_and_verify(submission['app_id'], environment=submission['environment'])
        write_json(args.directory / f"shutdown-{submission['attempt_id']}.json", result, replace=True)
        if args.action == 'cancel':
            write_json(args.directory / 'cancellation.json',
                       {'attempt_id': submission['attempt_id'], 'call_id': submission['call_id'],
                        'shutdown': result, 'actual_interactions': None}, replace=True)
    else:
        import modal
        try:
            state = _remote_state(experiment, submission)
            result = {'status': state['status'], 'call_completed': True,
                      'attempt_id': submission['attempt_id'],
                      'experiment_id': state['experiment_id'], 'nodes': state['nodes']}
        except TimeoutError:
            result = {'status': 'running', 'call_completed': False, 'attempt_id': submission['attempt_id']}
        write_json(args.directory / 'remote-status.json', result, replace=True)


if __name__ == '__main__':
    cli()
