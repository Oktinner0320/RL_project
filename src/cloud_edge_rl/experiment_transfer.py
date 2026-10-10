"""Bounded full experiment export, including isolated private artifacts; no compute."""
from __future__ import annotations

import argparse
import json
import stat
import sys
import time
import unicodedata
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
    max_entries: int = 250000
    max_file_bytes: int = 128 * 1024 * 1024
    max_total_bytes: int = 24 * 1024 * 1024 * 1024
    max_inventory_bytes: int = 64 * 1024 * 1024
    max_shard_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError('Graph transfer limits must be positive integers')
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError('Graph file limit exceeds total byte limit')
        caps = {'download_seconds': 7200, 'max_entries': 250000,
                'max_file_bytes': 128 * 1024**2, 'max_total_bytes': 24 * 1024**3,
                'max_inventory_bytes': 64 * 1024**2, 'max_shard_bytes': 128 * 1024**2}
        if any(getattr(self, key) > cap for key, cap in caps.items()):
            raise ValueError('Graph transfer limit exceeds hard export bound')


def _path(value):
    if (type(value) is not str or not value or '\\' in value or ':' in value
            or any(part in ('', '.', '..') for part in value.split('/'))
            or any(unicodedata.category(char) == 'Cc' for char in value)):
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
    # Fresh metadata per component; never reuse checks across export phases.
    for item in (path, *path.parents):
        try:
            metadata = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError('Symbolic public artifact path forbidden')
        # Reject all Windows reparse points, not only junctions.
        if getattr(metadata, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Reparse artifact path forbidden')
    return path


def _confined(root, relative):
    path = _check_local(root.joinpath(*_path(relative).parts))
    if not path.resolve().is_relative_to(root):
        raise ValueError('Experiment artifact escaped destination')
    return path


def _inventory(value, experiment_id, limits):
    if (type(value) is not dict or value.get('schema') not in ('experiment_inventory_v1', 'experiment_inventory_v2')
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
            if parts[2] in ('inventory.json', 'export', 'shards', 'tmp'):
                raise ValueError('Inventory cannot include export metadata')
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
    if value['schema'] == 'experiment_inventory_v2':
        shards = value.get('shards')
        if type(shards) is not list or not 1 <= len(shards) <= limits.max_entries:
            raise ValueError('Invalid bounded experiment shards')
        covered, shard_paths, archive_bytes = set(), set(), 0
        sizes = {row['path']: row['bytes'] for row in files}
        for shard in shards:
            if (type(shard) is not dict or set(shard) != {'path', 'bytes', 'members'}
                    or type(shard['bytes']) is not int or not 0 < shard['bytes'] <= limits.max_shard_bytes
                    or type(shard['members']) is not list or not shard['members']):
                raise ValueError('Invalid experiment shard/size')
            path = _path(shard['path'])
            if (path.parts[:3] != ('experiments', experiment_id, 'export')
                    or len(path.parts) != 4 or path.suffix != '.tar'
                    or shard['path'].casefold() in shard_paths):
                raise ValueError('Unsafe or duplicate experiment shard')
            shard_paths.add(shard['path'].casefold())
            payload = 0
            for member in shard['members']:
                if type(member) is not str or member not in seen or member in covered:
                    raise ValueError('Unlisted/duplicate experiment shard member')
                covered.add(member)
                payload += sizes[member]
            if payload > shard['bytes']:
                raise ValueError('Shard payload exceeds archive size')
            archive_bytes += shard['bytes']
        if covered != seen or archive_bytes > limits.max_total_bytes + limits.max_entries * 4096:
            raise ValueError('Incomplete or excessive experiment shards')
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
                       environment='main', worker=run_bounded, available_only=False,
                       expected_graph=None) -> dict:
    """Export the exact final inventory; success denotes transport, not semantic trust."""
    safe_run_id(experiment_id)
    limits = GraphTransferLimits() if limits is None else limits
    if type(limits) is not GraphTransferLimits:
        raise ValueError('Explicit graph transfer limits required')
    if type(available_only) is not bool:
        raise ValueError('Explicit available-only flag required')
    if available_only:
        from .experiment_graph import validate_graph
        if type(expected_graph) is not dict or expected_graph.get('schema') != 'experiment_graph_v1':
            raise ValueError('Exact frozen graph required for available-only download')
        validate_graph(expected_graph)
        _snapshot_ids(expected_graph, experiment_id)
    _check_local(destination)
    root = Path(destination).absolute()
    root = _confined(root, 'experiments').parent.resolve()
    stage = _confined(root, f'.experiment-transfers/{experiment_id}')
    stage.mkdir(parents=True, exist_ok=True)
    binding = {'volume_name': volume_name, 'environment': environment,
               'experiment_id': experiment_id, 'destination': str(root)}
    if available_only:
        binding['available_only'] = True
        binding['expected_graph'] = expected_graph
    binding_path = stage / 'binding.json'
    if binding_path.exists():
        if read_json(binding_path) != binding:
            raise ValueError('Experiment transfer binding changed on resume')
    else:
        write_json(binding_path, binding)
    completion_path = _check_local(stage / ('partial-complete.json' if available_only else 'complete.json'))
    completion_path.unlink(missing_ok=True)
    started = time.monotonic()
    common = [sys.executable, '-m', 'cloud_edge_rl.experiment_transfer', '--volume', volume_name,
              '--experiment-id', experiment_id, '--environment', environment,
              '--limits', json.dumps(asdict(limits))]
    inventory_path = stage / ('partial-inventory.json' if available_only else 'inventory.json')
    if not inventory_path.exists():
        try:
            worker([*common, '--action', 'available' if available_only else 'inventory',
                    '--output', str(inventory_path)],
                   min(limits.listing_seconds, limits.download_seconds),
                   stage / f'inventory-{uuid.uuid4().hex}.log')
        except Exception as error:
            if isinstance(error, TimeoutError):
                raise
            if available_only and not isinstance(error, FileNotFoundError):
                raise
            return {'experiment_id': experiment_id, 'transport_complete': False,
                    'semantic_verified': False,
                    'status': 'partial_not_available' if available_only else 'inventory_not_available',
                    'error': str(error), 'destination': str(root)}
    _no_symlinks(inventory_path)
    if inventory_path.stat().st_size > limits.max_inventory_bytes:
        raise ValueError('Experiment inventory document exceeds byte bound')
    inventory = read_json(inventory_path)
    if available_only and inventory == {'experiment_id': experiment_id, 'status': 'partial_not_available'}:
        inventory_path.unlink()
        return {'experiment_id': experiment_id, 'status': 'partial_not_available',
                'transport_complete': False, 'semantic_verified': False, 'destination': str(root)}
    if available_only and (type(inventory) is not dict
                           or inventory.get('schema') != 'experiment_inventory_v2'):
        raise ValueError('Pinned available export requires archive inventory v2')
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
    snapshot = read_json(snapshot_path)
    if available_only and snapshot != expected_graph:
        raise ValueError('Available experiment snapshot differs from exact frozen graph')
    node_ids, run_ids = _snapshot_ids(snapshot, experiment_id)
    for row in files:
        parts = PurePosixPath(row['path']).parts
        if parts[0] == 'runs' and parts[1] not in run_ids:
            raise ValueError('Inventory run is not bound by the experiment snapshot')
        if parts[0] == 'experiments' and parts[2] == 'nodes' and parts[3] not in node_ids:
            raise ValueError('Inventory node is not bound by the experiment snapshot')
    if inventory['schema'] == 'experiment_inventory_v2':
        by_path = {row['path']: row for row in files}
        for index, shard in enumerate(inventory['shards']):
            remaining = limits.download_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('Experiment transfer deadline; completed shards retained')
            target = _check_local(stage / PurePosixPath(shard['path']).name)
            receipt = _check_local(stage / f'shard-{index:05d}.receipt.json')
            row = {'path': shard['path'], 'name': target.name, 'bytes': shard['bytes']}
            if not _restore_file(row, target, receipt):
                worker([*common, '--action', 'read', '--remote-path', row['path'], '--bytes', str(row['bytes']),
                        '--output', str(target), '--receipt', str(receipt)],
                       min(limits.file_seconds, remaining), stage / f'shard-{index:05d}-{uuid.uuid4().hex}.log')
                if not _restore_file(row, target, receipt):
                    raise ValueError('Shard worker did not publish complete receipt')
            remaining = limits.download_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('Experiment transfer deadline before extraction')
            members = [{key: by_path[path][key] for key in ('path', 'bytes')} for path in shard['members']]
            # Thousands of atomic small-file publications need a larger bounded
            # worker slice than a single bulk network read; retain the 2h total.
            extract_seconds = min(remaining, max(limits.file_seconds,
                                                 min(1800, 30 + len(members) // 50)))
            member_path = _check_local(stage / f'shard-{index:05d}.members.json')
            write_json(member_path, members, replace=True)
            worker([*common, '--action', 'extract', '--output', str(target),
                    '--members', str(member_path), '--destination', str(root),
                    '--receipt', str(stage / f'shard-{index:05d}.extract.json'),
                    '--seconds', str(extract_seconds)], extract_seconds,
                   stage / f'extract-{index:05d}-{uuid.uuid4().hex}.log')
            extracted = read_json(stage / f'shard-{index:05d}.extract.json')
            if extracted != {'complete': True, 'members': members}:
                raise ValueError('Missing exact extraction receipt')
            write_json(progress_path, {'experiment_id': experiment_id, 'completed_shards': index + 1,
                                      'total_shards': len(inventory['shards']),
                                      'transport_complete': False}, replace=True)
            if available_only:
                print(f"Downloaded and extracted committed shard {index + 1}/{len(inventory['shards'])}",
                      flush=True)
        completed, completed_bytes = len(files), inventory['bytes']
    else:
        for index, row in enumerate(files):
            if index != snapshot_index:
                transfer(row, index)
    state = read_json(_confined(root, f'experiments/{experiment_id}/state.json'))
    if type(state) is not dict or state.get('experiment_id') != experiment_id:
        raise ValueError('Experiment state identity mismatch')
    if available_only:
        present_runs = {PurePosixPath(row['path']).parts[1] for row in files
                        if PurePosixPath(row['path']).parts[0] == 'runs'}
        result = {'experiment_id': experiment_id, 'status': 'partial_download',
                  'available_shards_complete': True, 'transport_complete': False,
                  'semantic_verified': False, 'files': completed, 'bytes': completed_bytes,
                  'shards': len(inventory['shards']), 'destination': str(root),
                  'inventory': str(inventory_path), 'missing_run_ids': sorted(run_ids - present_runs),
                  'elapsed_seconds': time.monotonic() - started}
        write_json(completion_path, result, replace=True)
        return result
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


def _available_inventory(volume, experiment_id, limits):
    """Pin only existing, receipted archives; never publish or start remote work."""
    prefix = f'experiments/{experiment_id}/export/'
    entries = {}
    for index, entry in enumerate(volume.iterdir(prefix.rstrip('/'), recursive=False)):
        if index >= 20000:
            raise ValueError('Available export listing exceeds 20000 entries')
        path = entry.path.lstrip('/')
        parsed = _path(path)
        if (not path.startswith(prefix) or len(parsed.parts) != 4
                or path.casefold() in entries or type(entry.size) is not int or entry.size < 0):
            raise ValueError('Invalid available export listing path/size')
        entries[path.casefold()] = (path, entry.size, int(entry.type))
    files, shards, receipt_bytes = [], [], 0
    for _, (path, size, kind) in sorted(entries.items()):
        if kind != 1 or not path.endswith('.tar'):
            continue
        receipt_path = path[:-4] + '.receipt.json'
        pair = entries.get(receipt_path.casefold())
        if pair is None:
            continue
        if pair[0] != receipt_path or pair[2] != 1:
            raise ValueError('Available archive receipt path/type mismatch')
        if not 0 < size <= limits.max_shard_bytes or not 0 < pair[1] <= 8 * 1024**2:
            raise ValueError('Available archive/receipt exceeds byte bound')
        receipt_bytes += pair[1]
        if receipt_bytes > 64 * 1024**2:
            raise ValueError('Available receipts exceed aggregate byte bound')
        data = bytearray()
        for chunk in volume.read_file(receipt_path):
            if type(chunk) is not bytes or len(data) + len(chunk) > pair[1]:
                raise ValueError('Available receipt stream exceeds listed size')
            data.extend(chunk)
        if len(data) != pair[1]:
            raise ValueError('Available receipt stream shorter than listed size')
        receipt = _decode_json(data.decode('utf-8'))
        if (type(receipt) is not dict or set(receipt) != {'binding', 'bytes'}
                or type(receipt['bytes']) is not int or receipt['bytes'] != size
                or type(receipt['binding']) is not dict
                or set(receipt['binding']) != {'path', 'members'}
                or receipt['binding']['path'] != path
                or type(receipt['binding']['members']) is not list
                or not receipt['binding']['members']):
            raise ValueError('Invalid available archive receipt binding/bytes')
        members = receipt['binding']['members']
        if len(files) + len(members) > limits.max_entries:
            raise ValueError('Available export exceeds file count bound')
        for row in members:
            if (type(row) is not dict or set(row) != {'path', 'bytes'}
                    or type(row['bytes']) is not int or not 0 <= row['bytes'] <= limits.max_file_bytes):
                raise ValueError('Invalid available receipt member')
            _path(row['path'])
        files.extend(members)
        shards.append({'path': path, 'bytes': size, 'members': [row['path'] for row in members]})
    required = {f'experiments/{experiment_id}/snapshot.json', f'experiments/{experiment_id}/state.json'}
    if not shards or not required <= {row['path'] for row in files}:
        raise FileNotFoundError('Partial export not available: completed snapshot/state base required')
    inventory = {'schema': 'experiment_inventory_v2', 'experiment_id': experiment_id,
                 'files': files, 'bytes': sum(row['bytes'] for row in files), 'shards': shards}
    _inventory(inventory, experiment_id, limits)
    return inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('inventory', 'available', 'read', 'extract'), required=True)
    parser.add_argument('--volume', required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--environment', default='main')
    parser.add_argument('--limits', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-path')
    parser.add_argument('--bytes', type=int)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--members', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--seconds', type=float)
    args = parser.parse_args(argv)
    safe_run_id(args.experiment_id)
    limits = GraphTransferLimits(**_decode_json(args.limits))
    _check_local(args.output)
    if args.action == 'extract':
        if (args.members is None or args.destination is None or args.receipt is None
                or args.seconds is None or not 0 < args.seconds <= limits.download_seconds):
            raise ValueError('Exact bounded extraction required')
        from .experiment_export import extract_shard
        extract_shard(args.output, read_json(_check_local(args.members)), args.destination,
                      args.receipt, seconds=args.seconds)
        return
    if args.action == 'read':
        path = _path(args.remote_path)
        if (not ((path.parts[0] == 'runs' and len(path.parts) >= 3)
                 or (path.parts[:2] == ('experiments', args.experiment_id) and len(path.parts) >= 3))
                or args.bytes is None or not 0 <= args.bytes <= (limits.max_shard_bytes
                    if path.parts[:3] == ('experiments', args.experiment_id, 'export') else limits.max_file_bytes)
                or args.receipt is None or args.output.name != path.name):
            raise ValueError('Exact bounded experiment read required')
        _check_local(args.receipt)
    import modal
    volume = modal.Volume.from_name(args.volume, environment_name=args.environment)
    if args.action == 'available':
        try:
            inventory = _available_inventory(volume, args.experiment_id, limits)
        except FileNotFoundError:
            write_json(args.output, {'experiment_id': args.experiment_id, 'status': 'partial_not_available'})
            return
        if len(json.dumps(inventory).encode('utf-8')) > limits.max_inventory_bytes:
            raise ValueError('Available inventory document exceeds byte bound')
        write_json(args.output, inventory)
    elif args.action == 'inventory':
        inventory = _read_inventory(volume.read_file(f'experiments/{args.experiment_id}/inventory.json'), limits)
        _inventory(inventory, args.experiment_id, limits)
        write_json(args.output, inventory)
    else:
        receive_file(volume.read_file(args.remote_path),
                     {'path': args.remote_path, 'name': path.name, 'bytes': args.bytes},
                     args.output, args.receipt)


if __name__ == '__main__':
    main()
