"""The single notebook-facing experiment lifecycle.

Planning is inert. Submission retains the existing admission and reservation
barriers. Cloud computation, artifact transfer and local closure are separate
durable states: a failed download can never start another training call.
"""
from __future__ import annotations

import argparse
import json
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

    def status(self, refresh=False):
        if refresh:
            self._command('status', 60)
        result = {'experiment_id': self.graph['experiment_id'],
                  'planned_nodes': len(self.graph['nodes']), 'status': 'prepared',
                  'formal_acceptance': 'not_evaluated', 'actual_cloud_cost_usd': None}
        for name in ('submission', 'remote-status', 'transfer', 'closure'):
            path = self.directory / f'{name}.json'
            if path.exists():
                result[name] = read_json(path)
        if 'closure' in result:
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
        shutdown = self.directory / 'shutdown.json'
        if not shutdown.exists():
            self._command('cleanup', 120)
        if read_json(shutdown).get('stopped') is not True:
            raise ValueError('Exact app shutdown not verified')
        result = collect_experiment(submission['volume_name'], self.graph['experiment_id'],
                                    self.directory / 'artifacts', environment=submission['environment'])
        write_json(self.directory / 'transfer.json', result, replace=True)
        return result

    def verify(self):
        """Entirely local closure; never opens Modal or starts compute."""
        artifacts = self.directory / 'artifacts'
        base = artifacts / 'experiments' / self.graph['experiment_id']
        if read_json(base / 'snapshot.json') != self.graph:
            raise ValueError('Returned graph differs from the frozen plan')
        state = read_json(base / 'state.json')
        if state['status'] != 'cloud_verified':
            raise ValueError('Full experiment has not reached cloud verification')
        if not (self.directory / 'transfer.json').exists():
            raise ValueError('Unified artifact transfer incomplete')
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
        result = {'experiment_id': self.graph['experiment_id'], 'status': 'completed',
                  'nodes': proofs, 'formal_acceptance': 'not_evaluated',
                  'actual_cloud_cost_usd': None}
        write_json(self.directory / 'closure.json', result, replace=True)
        return result


def run_remote_graph(graph, volume_name):
    """Detached call: commit progress independently of notebook connectivity."""
    import modal
    volume = modal.Volume.from_name(volume_name)
    volume.reload()
    return run_graph(graph, Path('/workspace'), commit=volume.commit)


def _submit(experiment, *, continuation=False):
    import modal
    from .admission import check_admission, reserve
    from .modal_runner import create_app

    graph, directory = experiment.graph, experiment.directory
    submission = directory / 'submission.json'
    lock = directory / 'submission.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(graph['experiment_id'])
    try:
        specs = [RunSpec(**expand(graph['pool'], node['spec_ref'])) for node in graph['nodes']]
        if continuation:
            prior = read_json(submission)
            result = modal.FunctionCall.from_id(prior['call_id']).get(timeout=0)
            if (result.get('experiment_id') != graph['experiment_id']
                    or result.get('status') not in ('paused_deadline', 'blocked')):
                raise ValueError('Only an observed paused or verification-blocked graph can continue')
            if any(row['status'] not in ('pending', 'verified', 'computed', 'verification_failed')
                   for row in result['nodes'].values()):
                raise ValueError('Ambiguous compute cannot be automatically retried')
            from .transport import stop_and_verify
            stopped = stop_and_verify(prior['app_id'], environment=prior['environment'])
            write_json(directory / f"prior-shutdown-{uuid.uuid4().hex}.json", stopped)
            write_json(directory / f"prior-submission-{uuid.uuid4().hex}.json", prior)
        elif submission.exists():
            raise ValueError('Existing submission cannot be repeated')
        # Preserve all historical approvals, allocations, failed attempts and source checks.
        check_admission(specs, experiment.root)
        if not continuation:
            for spec in specs:
                if reserve(spec, experiment.root) is not None:
                    raise ValueError('Legacy paid call must be recovered, never resubmitted')
        settings = specs[0].config['modal']
        record = {'experiment_id': graph['experiment_id'], 'status': 'reserved',
                  'call_id': None, 'app_id': None, 'volume_name': settings['volume_name'],
                  'environment': settings.get('environment', 'main'),
                  'reserved_run_ids': [s.run_id for s in specs],
                  'actual_cloud_cost_usd': None, 'historical_consumption_reset': False}
        write_json(submission, record, replace=continuation)
        app, function, _ = create_app(experiment.root, settings, specs,
                                      graph_timeout_seconds=graph['graph_seconds'])
        with modal.enable_output(), app.run(detach=True):
            record.update(status='submitting', app_id=app.app_id)
            write_json(submission, record, replace=True)
            call = function.spawn(graph, settings['volume_name'])
            record.update(status='submitted', call_id=call.object_id)
            write_json(submission, record, replace=True)
    finally:
        lock.unlink()


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('submit', 'resume', 'status', 'cleanup'), required=True)
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
    if args.action == 'cleanup':
        from .transport import stop_and_verify
        result = stop_and_verify(submission['app_id'], environment=submission['environment'])
        write_json(args.directory / 'shutdown.json', result)
    else:
        import modal
        try:
            state = modal.FunctionCall.from_id(submission['call_id']).get(timeout=0)
            if state.get('experiment_id') != experiment.graph['experiment_id']:
                raise ValueError('Remote experiment identity mismatch')
            result = {'status': state['status'], 'call_completed': True,
                      'experiment_id': state['experiment_id'], 'nodes': state['nodes']}
        except TimeoutError:
            result = {'status': 'running', 'call_completed': False}
        write_json(args.directory / 'remote-status.json', result, replace=True)


if __name__ == '__main__':
    cli()
