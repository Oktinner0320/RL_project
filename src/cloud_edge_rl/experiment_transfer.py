"""Bounded full experiment export, including isolated private artifacts; no compute."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .transport import (
    _decode_json,
    _no_symlinks,
    _restore_file,
    read_json,
    receive_file,
    run_bounded,
    safe_run_id,
    write_json,
)


@dataclass(frozen=True)
class GraphTransferLimits:
    listing_seconds: int = 45
    file_seconds: int = 90
    download_seconds: int = 7200
    max_entries: int = 20000
    max_file_bytes: int = 128 * 1024 * 1024
    max_total_bytes: int = 8 * 1024 * 1024 * 1024
    max_inventory_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError('Graph transfer limits must be positive integers')
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError('Graph file limit exceeds total byte limit')


def _path(value):
    if (type(value) is not str or not value or '\\' in value or ':' in value
            or any(part in ('', '.', '..') for part in value.split('/'))
            or any(ord(char) < 32 for char in value)):
        raise ValueError('Unsafe experiment artifact path')
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError('Noncanonical experiment artifact path')
    # Windows normalizes trailing dots/spaces and reserves these device filenames.
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                *(f'LPT{i}' for i in range(1, 10))}
    for part in path.parts:
        if (part.endswith(('.', ' ')) or part.split('.')[0].upper() in reserved
                or any(char in part for char in '<>"|?*')):
            raise ValueError('Nonportable experiment artifact path')
    return path


def _check_local(path):
    path = Path(path).absolute()
    _no_symlinks(path)
    # Junctions/reparse points can escape without Path.is_symlink() on Windows.
    for item in (path, *path.parents):
        if hasattr(item, 'is_junction') and item.is_junction():
            raise ValueError('Reparse artifact path forbidden')
        if item.exists() and getattr(item.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('Reparse artifact path forbidden')
    return path


def _confined(root, relative):
    path = _check_local(root.joinpath(*_path(relative).parts))
    if not path.resolve().is_relative_to(root):
        raise ValueError('Experiment artifact escaped destination')
    return path


def _inventory(value, experiment_id, limits):
    if (type(value) is not dict or value.get('schema') != 'experiment_inventory_v1'
            or value.get('experiment_id') != experiment_id or type(value.get('files')) is not list
            or not 2 <= len(value['files']) <= limits.max_entries
            or type(value.get('bytes')) is not int):
        raise ValueError('Invalid bounded experiment inventory')
    seen, folded, total, files = set(), set(), 0, []
    prefixes = {}
    for row in value['files']:
        if (type(row) is not dict or set(row) != {'path', 'bytes'}
                or type(row['bytes']) is not int or not 0 <= row['bytes'] <= limits.max_file_bytes):
            raise ValueError('Invalid experiment inventory file/size')
        path = _path(row['path'])
        parts = path.parts
        if parts[0] == 'runs' and len(parts) >= 3:
            safe_run_id(parts[1])
        elif parts[:2] == ('experiments', experiment_id) and len(parts) >= 3:
            if parts[2] == 'inventory.json':
                raise ValueError('Inventory cannot include itself')
            if parts[2] == 'nodes':
                if len(parts) < 5:
                    raise ValueError('Exact experiment node file required')
                safe_run_id(parts[3])
        else:
            raise ValueError('Inventory outside experiment/run roots')
        if row['path'] in seen or row['path'].casefold() in folded:
            raise ValueError('Duplicate/colliding experiment artifact path')
        for length in range(1, len(parts) + 1):
            prefix = '/'.join(parts[:length])
            if prefixes.setdefault(prefix.casefold(), prefix) != prefix:
                raise ValueError('Case-colliding experiment artifact directory')
        seen.add(row['path'])
        folded.add(row['path'].casefold())
        total += row['bytes']
        if total > limits.max_total_bytes:
            raise ValueError('Experiment inventory exceeds total byte bound')
        files.append({'path': row['path'], 'name': path.name, 'bytes': row['bytes']})
    # File/directory collisions are unsafe even on case-sensitive filesystems.
    for path in seen:
        if any(parent.as_posix().casefold() in folded for parent in PurePosixPath(path).parents):
            raise ValueError('Experiment inventory file/directory collision')
    required = {f'experiments/{experiment_id}/snapshot.json', f'experiments/{experiment_id}/state.json'}
    if total != value['bytes'] or not required <= seen:
        raise ValueError('Incomplete experiment inventory or byte total mismatch')
    return files


def _snapshot_ids(snapshot, experiment_id):
    if type(snapshot) is not dict or snapshot.get('experiment_id') != experiment_id:
        raise ValueError('Experiment snapshot identity mismatch')
    if snapshot.get('schema') == 'experiment_graph_v1':
        from .experiment_graph import expand, validate_graph
        validate_graph(snapshot)
        node_ids, run_ids = set(), set()
        folded = set()
        for node in snapshot['nodes']:
            run_id = node['run_id']
            spec = expand(snapshot['pool'], node['spec_ref'])
            if spec['run_id'] != run_id or run_id.casefold() in folded:
                raise ValueError('Graph snapshot pooled run identity mismatch/collision')
            safe_run_id(run_id)
            folded.add(run_id.casefold())
            node_ids.add(run_id)
            run_ids.add(run_id)
        return node_ids, run_ids
    nodes = snapshot.get('nodes')
    if type(nodes) is dict:
        items = list(nodes.items())
    elif type(nodes) is list:
        items = [(node.get('node_id') if type(node) is dict else None, node) for node in nodes]
    else:
        raise ValueError('Snapshot must declare exact nodes')
    node_ids, run_ids = set(), set()
    for key, node in items:
        if type(node) is not dict:
            raise ValueError('Invalid snapshot node')
        node_id = node.get('node_id', key)
        if node_id != key:
            raise ValueError('Snapshot node key mismatch')
        safe_run_id(node_id)
        spec = node.get('spec', node)
        if type(spec) is not dict:
            raise ValueError('Snapshot node must bind a run spec')
        run_id = spec.get('run_id')
        safe_run_id(run_id)
        if (node_id.casefold() in {value.casefold() for value in node_ids}
                or run_id.casefold() in {value.casefold() for value in run_ids}):
            raise ValueError('Duplicate snapshot node/run identity')
        node_ids.add(node_id)
        run_ids.add(run_id)
    if not node_ids:
        raise ValueError('Empty experiment snapshot')
    return node_ids, run_ids


def collect_experiment(volume_name, experiment_id, destination, *, limits=None,
                       environment='main', worker=run_bounded) -> dict:
    """Export the exact final inventory; success denotes transport, not semantic trust."""
    safe_run_id(experiment_id)
    limits = GraphTransferLimits() if limits is None else limits
    if type(limits) is not GraphTransferLimits:
        raise ValueError('Explicit graph transfer limits required')
    _check_local(destination)
    root = Path(destination).absolute()
    root = _confined(root, 'experiments').parent.resolve()
    stage = _confined(root, f'.experiment-transfers/{experiment_id}')
    stage.mkdir(parents=True, exist_ok=True)
    binding = {'volume_name': volume_name, 'environment': environment,
               'experiment_id': experiment_id, 'destination': str(root)}
    binding_path = stage / 'binding.json'
    if binding_path.exists():
        if read_json(binding_path) != binding:
            raise ValueError('Experiment transfer binding changed on resume')
    else:
        write_json(binding_path, binding)
    completion_path = _check_local(stage / 'complete.json')
    completion_path.unlink(missing_ok=True)
    started = time.monotonic()
    common = [sys.executable, '-m', 'cloud_edge_rl.experiment_transfer', '--volume', volume_name,
              '--experiment-id', experiment_id, '--environment', environment,
              '--limits', json.dumps(asdict(limits))]
    inventory_path = stage / 'inventory.json'
    if not inventory_path.exists():
        try:
            worker([*common, '--action', 'inventory', '--output', str(inventory_path)],
                   min(limits.listing_seconds, limits.download_seconds),
                   stage / f'inventory-{uuid.uuid4().hex}.log')
        except Exception as error:
            if isinstance(error, TimeoutError):
                raise
            return {'experiment_id': experiment_id, 'transport_complete': False,
                    'semantic_verified': False, 'status': 'inventory_not_available',
                    'error': str(error), 'destination': str(root)}
    _no_symlinks(inventory_path)
    if inventory_path.stat().st_size > limits.max_inventory_bytes:
        raise ValueError('Experiment inventory document exceeds byte bound')
    inventory = read_json(inventory_path)
    files = _inventory(inventory, experiment_id, limits)
    progress_path = stage / 'progress.json'
    completed, completed_bytes = 0, 0

    def transfer(row, index):
        nonlocal completed, completed_bytes
        target = _confined(root, row['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = stage / f'file-{index:05d}.receipt.json'
        _no_symlinks(receipt)
        remaining = limits.download_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Experiment transfer deadline; completed files retained')
        if not _restore_file(row, target, receipt):
            worker([*common, '--action', 'read', '--remote-path', row['path'], '--bytes', str(row['bytes']),
                    '--output', str(target), '--receipt', str(receipt)],
                   min(limits.file_seconds, remaining), stage / f'read-{index:05d}-{uuid.uuid4().hex}.log')
            if not _restore_file(row, target, receipt):
                raise ValueError('Experiment worker did not publish a complete receipt')
        completed += 1
        completed_bytes += row['bytes']
        write_json(progress_path, {'experiment_id': experiment_id, 'completed_files': completed,
                                  'completed_bytes': completed_bytes, 'total_files': len(files),
                                  'total_bytes': inventory['bytes'], 'last_path': row['path'],
                                  'transport_complete': False}, replace=True)
        return target

    snapshot_remote = f'experiments/{experiment_id}/snapshot.json'
    snapshot_index = next(index for index, row in enumerate(files) if row['path'] == snapshot_remote)
    snapshot_path = transfer(files[snapshot_index], snapshot_index)
    node_ids, run_ids = _snapshot_ids(read_json(snapshot_path), experiment_id)
    for row in files:
        parts = PurePosixPath(row['path']).parts
        if parts[0] == 'runs' and parts[1] not in run_ids:
            raise ValueError('Inventory run is not bound by the experiment snapshot')
        if parts[0] == 'experiments' and parts[2] == 'nodes' and parts[3] not in node_ids:
            raise ValueError('Inventory node is not bound by the experiment snapshot')
    for index, row in enumerate(files):
        if index != snapshot_index:
            transfer(row, index)
    state = read_json(_confined(root, f'experiments/{experiment_id}/state.json'))
    if type(state) is not dict or state.get('experiment_id') != experiment_id:
        raise ValueError('Experiment state identity mismatch')
    # A failed graph may legitimately lack results for unstarted/failed nodes.
    # A globally completed graph must retain every result and native gate receipt.
    if state.get('status', state.get('state')) in ('completed', 'cloud_verified'):
        paths = {row['path'] for row in files}
        for node_id in node_ids:
            prefix = f'experiments/{experiment_id}/nodes/{node_id}'
            if not {prefix + '/result.json', prefix + '/verification.json'} <= paths:
                raise ValueError('Completed experiment missing node result/verification artifacts')
    # Retain the authoritative inventory alongside its downloaded experiment files.
    exported_inventory = _confined(root, f'experiments/{experiment_id}/inventory.json')
    if exported_inventory.exists():
        if read_json(exported_inventory) != inventory:
            raise ValueError('Exported experiment inventory changed')
    else:
        write_json(exported_inventory, inventory)
    result = {'experiment_id': experiment_id, 'transport_complete': True,
              'status': 'transport_complete', 'graph_status': state.get('status', state.get('state')),
              'semantic_verified': False, 'files': completed, 'bytes': completed_bytes,
              'destination': str(root), 'inventory': str(exported_inventory),
              'elapsed_seconds': time.monotonic() - started}
    write_json(completion_path, result, replace=True)
    return result


def _read_inventory(chunks, limits):
    data = bytearray()
    for chunk in chunks:
        if type(chunk) is not bytes or len(data) + len(chunk) > limits.max_inventory_bytes:
            raise ValueError('Experiment inventory stream exceeds byte bound')
        data.extend(chunk)
    return _decode_json(data.decode('utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('inventory', 'read'), required=True)
    parser.add_argument('--volume', required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--environment', default='main')
    parser.add_argument('--limits', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-path')
    parser.add_argument('--bytes', type=int)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args(argv)
    safe_run_id(args.experiment_id)
    limits = GraphTransferLimits(**_decode_json(args.limits))
    _check_local(args.output)
    if args.action == 'read':
        path = _path(args.remote_path)
        if (not ((path.parts[0] == 'runs' and len(path.parts) >= 3)
                 or (path.parts[:2] == ('experiments', args.experiment_id) and len(path.parts) >= 3))
                or args.bytes is None or not 0 <= args.bytes <= limits.max_file_bytes
                or args.receipt is None or args.output.name != path.name):
            raise ValueError('Exact bounded experiment read required')
        _check_local(args.receipt)
    import modal
    volume = modal.Volume.from_name(args.volume, environment_name=args.environment)
    if args.action == 'inventory':
        inventory = _read_inventory(volume.read_file(f'experiments/{args.experiment_id}/inventory.json'), limits)
        _inventory(inventory, args.experiment_id, limits)
        write_json(args.output, inventory)
    else:
        receive_file(volume.read_file(args.remote_path),
                     {'path': args.remote_path, 'name': path.name, 'bytes': args.bytes},
                     args.output, args.receipt)


if __name__ == '__main__':
    main()
