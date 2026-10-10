"""Export already verified artifacts on CPU; never execute the experiment graph."""
from __future__ import annotations

from pathlib import Path

from .experiment_export import publish_shards
from .experiment_transfer import _check_local
from .transport import read_json, safe_run_id, write_json

_BINDING_KEYS = {'experiment_id', 'attempt_id', 'call_id', 'app_id',
                 'volume_name', 'environment'}


def _validate_binding(graph, binding):
    if type(binding) is not dict or set(binding) != _BINDING_KEYS:
        raise ValueError('Exact original export recovery binding required')
    for value in binding.values():
        if type(value) is not str or len(value) > 256:
            raise ValueError('Bounded export recovery binding required')
        safe_run_id(value)
    if type(graph) is not dict or graph.get('experiment_id') != binding['experiment_id']:
        raise ValueError('Export recovery experiment identity mismatch')


def validate_exact_completed_graph(graph, workspace, binding):
    """Read-only gate against the frozen graph and original export-failed receipt.

    Original SDK call/app identity is reconciled by the controller; the remote
    graph receipt stores the attempt identity, not SDK call/app identifiers.
    """
    _validate_binding(graph, binding)
    workspace = _check_local(workspace)
    base = _check_local(workspace / 'experiments' / binding['experiment_id'])
    snapshot = read_json(_check_local(base / 'snapshot.json'))
    if snapshot != graph:
        raise ValueError('Export recovery snapshot differs from frozen graph')
    nodes = graph.get('nodes')
    if type(nodes) is not list or not nodes:
        raise ValueError('Export recovery requires a nonempty frozen graph')
    run_ids = []
    for node in nodes:
        if type(node) is not dict:
            raise ValueError('Invalid frozen graph node')
        run_ids.append(safe_run_id(node.get('run_id')))
    if len(set(run_ids)) != len(run_ids):
        raise ValueError('Duplicate frozen graph run identity')
    state = read_json(_check_local(base / 'state.json'))
    receipt = read_json(_check_local(base / 'calls' / f"{binding['attempt_id']}.json"))
    if type(receipt) is not dict or receipt.get('attempt_id') != binding['attempt_id']:
        raise ValueError('Original export receipt attempt mismatch')
    original = receipt.get('state')
    if (type(original) is not dict or original.get('status') != 'blocked'
            or type(original.get('error')) is not dict
            or original['error'].get('phase') != 'export'):
        raise ValueError('Original call did not fail only in export')
    for value in (state, original):
        if (type(value) is not dict or value.get('experiment_id') != binding['experiment_id']
                or type(value.get('nodes')) is not dict or set(value['nodes']) != set(run_ids)
                or any(type(row) is not dict or row.get('status') != 'verified'
                       for row in value['nodes'].values())):
            raise ValueError('Export recovery requires every exact graph node verified')
    if state.get('status') != 'cloud_verified':
        raise ValueError('Persisted experiment is not cloud_verified')
    if original['nodes'] != state['nodes']:
        raise ValueError('Original export receipt is stale relative to cloud state')
    return state


def run_export_recovery(graph, volume_name, binding, recovery_id):
    """One bounded export-only call; controller owns reservation and app cleanup."""
    _validate_binding(graph, binding)
    safe_run_id(recovery_id)
    if len(recovery_id) > 128 or volume_name != binding['volume_name']:
        raise ValueError('Export recovery volume or identity mismatch')
    import modal

    volume = modal.Volume.from_name(volume_name, environment_name=binding['environment'])
    volume.reload()
    # Only this fixed trusted mount root may be canonicalized before validation.
    # Every artifact descendant is still checked in its original lexical form.
    workspace = _check_local(Path('/workspace').resolve(strict=True))
    receipt = _check_local(workspace / 'experiments' / binding['experiment_id']
                           / 'calls' / f'export-{recovery_id}.json')
    if receipt.exists():
        raise ValueError('Export recovery identity already has a receipt; reconcile before retry')
    result = {**binding, 'recovery_id': recovery_id, 'status': 'blocked'}
    try:
        validate_exact_completed_graph(graph, workspace, binding)
        inventory = publish_shards(graph, workspace, commit=volume.commit)
        result.update(status='export_complete', files=len(inventory['files']),
                      bytes=inventory['bytes'], shards=len(inventory['shards']))
    except Exception as error:
        # Never embed the large graph or source evidence in the controller receipt.
        result['error'] = {'phase': 'export', 'type': type(error).__name__,
                           'message': str(error)[:4096]}
    write_json(receipt, result)
    volume.commit()
    return result


def create_export_recovery_app(volume_name, environment):
    """Construct, but do not open, a minimal existing-volume CPU export app."""
    safe_run_id(volume_name)
    safe_run_id(environment)
    import modal

    image = modal.Image.debian_slim(python_version='3.12').add_local_python_source('cloud_edge_rl')
    volume = modal.Volume.from_name(volume_name, environment_name=environment)
    app = modal.App('experiment-export-recovery')
    function = app.function(
        name='run_export_recovery', image=image, volumes={'/workspace': volume},
        cpu=2, memory=2048, max_containers=1, min_containers=0, buffer_containers=0,
        retries=0, timeout=1800, startup_timeout=120,
    )(run_export_recovery)
    return app, function
