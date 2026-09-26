"""Inspect a distributed resume and optionally remove caches outside its active round."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .collection_lifecycle import (
    expected_round_manifest_path,
    remove_completed_collection,
    validate_active_collection,
    validate_round_manifest_path,
)
from .distributed_train import _pooled_manifest_root, _validate_resume_distributed_config
from .runtime import load_config, resolve_path


def plan_resume(config: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all rank cursors and manifests before proposing any removal."""
    if payload.get('format') != 'condition_b_v1' or not payload.get('optimizer_state'):
        raise ValueError('Resume requires a Condition B checkpoint with optimizer state')
    distributed = payload.get('distributed_state', {})
    world_size = config['training']['world_size']
    if distributed.get('world_size') != world_size:
        raise ValueError('Checkpoint world size differs from the saved run configuration')
    _validate_resume_distributed_config(config, payload['config'], world_size=world_size)
    rank_states = distributed.get('rank_states')
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise ValueError('Checkpoint is missing rank states')
    cleanup = []
    cursors = []
    run_id = config['training']['run_id']
    for rank_idx, rank_state in enumerate(rank_states):
        if rank_state.get('rank') != rank_idx or not rank_state.get('rng_state'):
            raise ValueError(f'Checkpoint is missing rank {rank_idx} state/RNG')
        sampler = rank_state['sampler_state']
        round_idx = sampler['collection_round_idx']
        if type(round_idx) is not int or not 0 <= round_idx <= config['collection']['num_collections']:
            raise ValueError('Checkpoint collection cursor is out of range')
        training_root = _pooled_manifest_root(config, run_id, rank_idx) / 'train'
        active_path = sampler.get('manifest_path')
        if active_path:
            manifest_path = validate_round_manifest_path(training_root, round_idx, active_path)
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('complete') is not True or manifest.get('split') != 'train':
                raise ValueError(f'Active training manifest is incomplete: {manifest_path}')
            if manifest.get('teacher_checkpoint_sha256') != config['teacher_checkpoint_sha256']:
                raise ValueError(f'Active collection teacher differs: {manifest_path}')
        pending = sampler.get('pending_cleanup_round_idx')
        if pending is not None and (type(pending) is not int or pending != round_idx - 1):
            raise ValueError('Invalid pending cleanup cursor')
        for directory in sorted(training_root.glob('round_*')):
            suffix = directory.name.removeprefix('round_')
            if not suffix.isdigit() or directory.name != f'round_{int(suffix):04d}':
                raise ValueError(f'Unexpected collection directory: {directory}')
            stored_round_idx = int(suffix)
            if stored_round_idx == round_idx and active_path:
                continue
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(f'Unsafe collection directory: {directory}')
            if any(path.is_symlink() for path in directory.rglob('*')):
                raise ValueError(f'Collection contains a symlink: {directory}')
            # Resolve every deletion candidate now, before the first removal.
            expected_round_manifest_path(training_root, stored_round_idx)
            allocated_bytes = sum(path.stat().st_blocks * 512 for path in directory.rglob('*') if path.is_file())
            cleanup.append({'training_root': str(training_root), 'round_idx': stored_round_idx,
                            'path': str(directory), 'allocated_bytes': allocated_bytes,
                            'reason': 'completed before checkpoint' if stored_round_idx < round_idx
                            else 'experience not referenced by checkpoint; recollect on resume'})
        cursors.append({'rank': rank_idx, 'collection_round_idx': round_idx,
                        'update_epoch_idx': sampler.get('update_epoch_idx', 0),
                        'next_batch_idx': sampler.get('next_batch_idx', 0),
                        'active_manifest': str(active_path) if active_path else None})
    if len({(row['collection_round_idx'], row['update_epoch_idx'], row['next_batch_idx']) for row in cursors}) != 1:
        raise ValueError('Distributed checkpoint cursors disagree')
    validation = payload.get('collection_state', {}).get('validation_manifest')
    if not validation or not resolve_path(validation).is_file():
        raise FileNotFoundError('Checkpoint validation manifest is missing')
    return {'optimizer_step': payload['step'], 'world_size': world_size, 'rank_cursors': cursors,
            'validation_manifest': validation, 'cleanup': cleanup,
            'cleanup_allocated_gib': sum(item['allocated_bytes'] for item in cleanup) / 2**30}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cleanup-cached', action='store_true',
                        help='Delete listed past/uncheckpointed rounds; preserve the checkpoint active round and validation')
    parser.add_argument('--require-ready', action='store_true')
    arguments = parser.parse_args()
    import torch

    config = load_config(arguments.config)
    payload = torch.load(resolve_path(arguments.checkpoint), map_location='cpu', weights_only=False)
    plan = plan_resume(config, payload)
    print(json.dumps(plan, indent=2))
    if arguments.cleanup_cached:
        for item in plan['cleanup']:
            remove_completed_collection(Path(item['training_root']), item['round_idx'])
    if arguments.require_ready or arguments.cleanup_cached:
        if plan['cleanup'] and not arguments.cleanup_cached:
            raise RuntimeError('Extra cached rounds remain. Review the plan, then launch with CLEANUP_CACHED=1.')
        for cursor in plan['rank_cursors']:
            root = _pooled_manifest_root(config, config['training']['run_id'], cursor['rank']) / 'train'
            validate_active_collection(root, cursor['collection_round_idx'] if cursor['active_manifest'] else None)


if __name__ == '__main__':
    main()
