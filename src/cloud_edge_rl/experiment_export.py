"""Bounded, uncompressed experiment shards; frozen bytes, never model semantics."""
from __future__ import annotations

import os
import tarfile
import time
import uuid
from pathlib import Path

from .transport import read_json, write_json

SHARD_BYTES = 128 * 1024**2
EXPORT_SECONDS = 1800


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Experiment export deadline; committed shards retained')


def _equal(left, right, deadline):
    while True:
        _check_deadline(deadline)
        a, b = left.read(1024 * 1024), right.read(1024 * 1024)
        if a != b:
            return False
        if not a:
            return True


def _info(row):
    info = tarfile.TarInfo(row['path'])
    info.size = row['bytes']
    info.mode = 0o600
    return info


def _matches(archive, rows, workspace, deadline):
    from .experiment_transfer import _confined
    with tarfile.open(archive, 'r:') as tar:
        for row in rows:
            _check_deadline(deadline)
            member = tar.next()
            if (member is None or not member.isfile() or member.name != row['path']
                    or member.size != row['bytes']):
                return False
            with tar.extractfile(member) as frozen, _confined(workspace, row['path']).open('rb') as source:
                if not _equal(frozen, source, deadline):
                    return False
        return tar.next() is None


def publish_shards(graph, workspace, *, commit=lambda: None, seconds=EXPORT_SECONDS):
    from .experiment_transfer import GraphTransferLimits, _check_local, _inventory
    if not 0 < seconds <= EXPORT_SECONDS:
        raise ValueError('Bounded export deadline required')
    deadline = time.monotonic() + seconds
    workspace = _check_local(workspace).resolve()
    experiment_id = graph['experiment_id']
    base = workspace / 'experiments' / experiment_id
    output = _check_local(base / 'export')
    output.mkdir(parents=True, exist_ok=True)
    limits = GraphTransferLimits()
    groups, rows, total = [], [], 0
    owners = [('base', base), *((n['run_id'], workspace / 'runs' / n['run_id']) for n in graph['nodes'])]
    for label, owner in owners:
        batch, used, index = [], 10240, 0
        if not owner.exists():
            continue
        _check_local(owner)
        # Do not traverse the export directory, including abandoned shard partials.
        for directory, dirs, names in os.walk(owner, followlinks=False):
            _check_deadline(deadline)
            directory = Path(directory)
            for name in list(dirs):
                child = directory / name
                _check_local(child)
                if child == output:
                    dirs.remove(name)
            dirs.sort()
            for name in sorted(names):
                path = _check_local(directory / name)
                if path == base / 'inventory.json' or name.endswith('.part'):
                    continue
                if not path.is_file():
                    raise ValueError('Nonregular experiment artifact')
                row = {'path': path.relative_to(workspace).as_posix(), 'bytes': path.stat().st_size}
                size = len(_info(row).tobuf(format=tarfile.PAX_FORMAT)) + ((row['bytes'] + 511) // 512) * 512
                if size + 10240 > SHARD_BYTES:
                    raise ValueError('Experiment source cannot fit bounded tar shard')
                if batch and used + size > SHARD_BYTES:
                    groups.append((label, index, batch))
                    batch, used, index = [], 10240, index + 1
                batch.append(row)
                used += size
                rows.append(row)
                total += row['bytes']
                if len(rows) > limits.max_entries or total > limits.max_total_bytes:
                    raise ValueError('Whole-experiment transfer inventory exceeds declared limits')
        if batch:
            groups.append((label, index, batch))
    value = {'schema': 'experiment_inventory_v2', 'experiment_id': experiment_id,
             'files': rows, 'bytes': total, 'shards': []}
    # Validate original paths before opening or publishing any archive.
    _inventory({**value, 'schema': 'experiment_inventory_v1'}, experiment_id, limits)
    for label, index, members in groups:
        _check_deadline(deadline)
        archive = _check_local(output / f'{label}-{index:05d}.tar')
        receipt = _check_local(archive.with_suffix('.receipt.json'))
        binding = {'members': members, 'path': archive.relative_to(workspace).as_posix()}
        reusable = (archive.exists() and receipt.exists() and read_json(receipt).get('binding') == binding
                    and read_json(receipt).get('bytes') == archive.stat().st_size
                    and _matches(archive, members, workspace, deadline))
        if not reusable:
            partial = archive.with_name(f'.{archive.name}.{uuid.uuid4().hex}.part')
            with partial.open('xb') as stream:
                with tarfile.open(fileobj=stream, mode='w', format=tarfile.PAX_FORMAT) as tar:
                    for row in members:
                        _check_deadline(deadline)
                        source = _check_local(workspace / row['path'])
                        with source.open('rb') as file:
                            tar.addfile(_info(row), file)
                stream.flush()
                os.fsync(stream.fileno())
            if partial.stat().st_size > SHARD_BYTES or not _matches(partial, members, workspace, deadline):
                raise ValueError('Source changed during experiment export')
            os.replace(partial, archive)
            write_json(receipt, {'binding': binding, 'bytes': archive.stat().st_size}, replace=True)
            commit()
        value['shards'].append({'path': binding['path'], 'bytes': archive.stat().st_size,
                                'members': [row['path'] for row in members]})
    _inventory(value, experiment_id, limits)
    write_json(base / 'inventory.json', value, replace=True)
    if (base / 'inventory.json').stat().st_size > limits.max_inventory_bytes:
        (base / 'inventory.json').unlink()
        raise ValueError('Experiment inventory document exceeds byte bound')
    commit()
    return value


def extract_shard(archive, rows, root, receipt, *, seconds):
    """Publish regular members atomically; interrupted extraction compares frozen bytes."""
    from .experiment_transfer import _check_local, _confined
    deadline = time.monotonic() + seconds
    root = _check_local(root).resolve()
    archive, receipt = _check_local(archive), _check_local(receipt)
    with tarfile.open(archive, 'r:') as tar:
        for row in rows:
            _check_deadline(deadline)
            member = tar.next()
            if (member is None or not member.isfile() or member.name != row['path']
                    or member.size != row['bytes']):
                raise ValueError('Tar member disagrees with exact inventory')
            target = _confined(root, row['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source:
                if target.exists():
                    with target.open('rb') as existing:
                        if not _equal(source, existing, deadline):
                            raise ValueError('Existing artifact differs from frozen shard')
                    continue
                partial = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.part')
                with partial.open('xb') as stream:
                    remaining = row['bytes']
                    while remaining:
                        _check_deadline(deadline)
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError('Truncated tar member')
                        stream.write(chunk)
                        remaining -= len(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(partial, target)
                partial.unlink()
        if tar.next() is not None:
            raise ValueError('Unlisted tar member')
    write_json(receipt, {'complete': True, 'members': rows}, replace=True)
