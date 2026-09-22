"""Shared configuration and environment construction for Condition B commands."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def merge_config(base: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested mappings without mutating the caller or recursive calls."""
    result = copy.deepcopy(dict(base))
    pending = [(result, changes)]
    # Each visited mapping is an input node; reject unreasonable config depth/size.
    for _ in range(10000):
        if not pending:
            return result
        destination, source = pending.pop()
        for key, value in source.items():
            if isinstance(value, Mapping) and isinstance(destination.get(key), dict):
                pending.append((destination[key], value))
            else:
                destination[key] = copy.deepcopy(value)
    raise ValueError('Configuration contains more than 10000 nested mappings')


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Read YAML, optional one-level extends, and explicit dotted KEY=VALUE overrides."""
    path = resolve_path(path)
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f'Configuration must be a mapping: {path}')
    parent = config.pop('extends', None)
    if parent is not None:
        parent_path = path.parent / parent
        with parent_path.open() as stream:
            inherited = yaml.safe_load(stream)
        if not isinstance(inherited, dict) or 'extends' in inherited:
            raise ValueError('extends must reference a mapping without another extends')
        config = merge_config(inherited, config)
    for override in overrides or []:
        if '=' not in override:
            raise ValueError(f'Expected KEY=VALUE override: {override}')
        key, raw_value = override.split('=', 1)
        parts = key.split('.')
        destination = config
        for part in parts[:-1]:
            if not isinstance(destination.get(part), dict):
                raise ValueError(f'Unknown configuration group in {key}')
            destination = destination[part]
        if parts[-1] not in destination:
            raise ValueError(f'Unknown configuration option: {key}')
        destination[parts[-1]] = yaml.safe_load(raw_value)
    return config


def create_vecenv(teacher_config: Mapping[str, Any], config: Mapping[str, Any], split: str):
    """Reuse PufferLib CPU workers with a full synchronous inference batch."""
    import pufferlib.vector
    from pufferlib.ocean.drive.drive import Drive

    vector = dict(config['vec'])
    count = vector['num_envs']
    workers = vector['num_workers']
    if not isinstance(count, int) or count < 1 or not isinstance(workers, int) or workers < 1:
        raise ValueError('vec.num_envs and vec.num_workers must be positive integers')
    if count % workers:
        raise ValueError('vec.num_envs must be divisible by vec.num_workers')
    if vector['batch_size'] != count:
        raise ValueError('Condition B currently requires vec.batch_size == vec.num_envs')
    seed = int(config['collection']['split_seeds'][split])
    return pufferlib.vector.make(
        Drive, env_kwargs=dict(teacher_config['env']), backend=vector['backend'],
        num_envs=count, num_workers=workers, batch_size=count,
        zero_copy=True, seed=seed,
    )


def prepare_runtime(config: Mapping[str, Any]) -> None:
    """Bound CPU threads and seed all learner RNGs before initialization."""
    import random
    import numpy as np
    import torch

    threads = int(config['training'].get('cpu_threads', 1))
    if threads < 1:
        raise ValueError('training.cpu_threads must be positive')
    torch.set_num_threads(threads)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(threads)
    seed = int(config['training']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config['training']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; set training.device=cpu explicitly')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
