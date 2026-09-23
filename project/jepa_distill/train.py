"""Repeated teacher collection, student updates, and checkpoint declarations."""

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, TYPE_CHECKING, Union

import copy
import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn

from .dataset import TrainingBatch
from .evaluate import evaluate_student
from .model import LossTerms

if TYPE_CHECKING:
    from pufferlib.ocean.drive.drive import Drive
    from .monitoring import WandbMonitor


PathLike = Union[str, Path]


def validate_driving_evaluation_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the optional periodic student driving evaluation protocol."""
    section = config.get('driving_evaluation')
    if section is None:
        return {'enabled': False}
    if not isinstance(section, Mapping):
        raise ValueError('driving_evaluation must be a mapping')
    enabled = section.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('driving_evaluation.enabled must be a boolean')
    if not enabled:
        return section

    for name in ('interval_steps', 'num_scenarios', 'episode_timesteps'):
        value = section.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'driving_evaluation.{name} must be a positive integer')

    population = section.get('population', 'student_self_play')
    if population != 'student_self_play':
        raise ValueError(
            "driving_evaluation.population must be 'student_self_play'; "
            f'got {population!r}'
        )
    execution_horizon = section.get('execution_horizon', 1)
    if isinstance(execution_horizon, bool) or not isinstance(execution_horizon, int) or execution_horizon != 1:
        raise ValueError('driving_evaluation.execution_horizon must be the integer 1')
    action_selection = section.get('action_selection', 'mean')
    if action_selection not in ('mean', 'mode', 'sample'):
        raise ValueError(
            'driving_evaluation.action_selection must be mean, mode, or sample'
        )
    benchmarks = section.get('benchmarks', ['carla'])
    if isinstance(benchmarks, str):
        benchmarks = [item.strip() for item in benchmarks.split(',') if item.strip()]
    if not isinstance(benchmarks, (list, tuple)) or not benchmarks:
        raise ValueError('driving_evaluation.benchmarks must select at least one benchmark')
    if any(not isinstance(name, str) or not name.strip() for name in benchmarks):
        raise ValueError('driving_evaluation.benchmarks must contain non-empty strings')
    for name in ('env_overrides', 'vec_overrides'):
        value = section.get(name)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f'driving_evaluation.{name} must be a mapping')
    if section.get('vec') is not None and not isinstance(section.get('vec'), Mapping):
        raise ValueError('driving_evaluation.vec must be a mapping')
    if section.get('num_agents') is not None:
        value = section['num_agents']
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('driving_evaluation.num_agents must be a positive integer')
    seed = section.get('seed')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1):
        raise ValueError('driving_evaluation.seed must be an integer in [0, 2147483647] or null')
    transfer = section.get('transfer')
    if transfer is not None:
        if not isinstance(transfer, Mapping):
            raise ValueError('driving_evaluation.transfer must be a mapping')
        transfer_enabled = transfer.get('enabled', False)
        if not isinstance(transfer_enabled, bool):
            raise ValueError('driving_evaluation.transfer.enabled must be a boolean')
        if transfer_enabled:
            raise ValueError('driving_evaluation.transfer.enabled=true is unsupported during training')
    for name in ('amp', 'compile'):
        if name in section and not isinstance(section[name], bool):
            raise ValueError(f'driving_evaluation.{name} must be a boolean')
        if section.get(name, False):
            raise ValueError(f'driving_evaluation.{name}=true is unsupported during training')
    output_name = section.get('output_name', 'training_native')
    if not isinstance(output_name, str) or not output_name.strip():
        raise ValueError('driving_evaluation.output_name must be a non-empty string')
    return section


def _capture_evaluation_state(student: nn.Module) -> tuple[dict[str, Any], list[tuple[nn.Module, bool]]]:
    """Capture every RNG stream and every module flag before simulator evaluation."""
    numpy_state = np.random.get_state()
    rng_state = {
        'python': random.getstate(),
        'numpy': (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        'torch': torch.get_rng_state().clone(),
        'cuda': [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }
    module_flags = [(module, bool(module.training)) for module in student.modules()]
    return rng_state, module_flags


def _restore_evaluation_state(
    rng_state: Mapping[str, Any], module_flags: Iterable[tuple[nn.Module, bool]]
) -> None:
    """Restore state without calling ``Module.train`` recursively."""
    for module, was_training in module_flags:
        # Direct assignment preserves nested EMA or custom module flags exactly.
        module.training = was_training
    torch.set_rng_state(rng_state['torch'])
    random.setstate(rng_state['python'])
    np.random.set_state(rng_state['numpy'])
    if rng_state['cuda'] is not None:
        torch.cuda.set_rng_state_all(rng_state['cuda'])


def _evaluation_output_subdir(
    run_dir: Path, evaluation_config: Mapping[str, Any], step: int
) -> str:
    """Choose a fresh report directory after an interrupted evaluation."""
    benchmark_names = evaluation_config.get('benchmarks', ['carla'])
    if isinstance(benchmark_names, str):
        benchmark_names = [item.strip() for item in benchmark_names.split(',') if item.strip()]
    output_name = str(evaluation_config.get('output_name', 'training_native'))
    base_name = f'step_{step:06d}'
    for attempt in range(10000):
        suffix = '' if attempt == 0 else f'_retry_{attempt:02d}'
        candidate_name = f'training/{base_name}{suffix}'
        occupied = any(
            (run_dir / 'eval' / f'{benchmark}_{output_name}' / candidate_name).exists()
            for benchmark in benchmark_names
        )
        if not occupied:
            return candidate_name
    raise RuntimeError(f'Unable to allocate a driving evaluation output directory for step {step}')


def _compact_driving_evaluation_results(
    results: Mapping[str, Any], benchmark_names: Iterable[str]
) -> dict[str, Any]:
    """Keep checkpoint/result metadata bounded to benchmark summaries."""
    compact: dict[str, Any] = {}
    for benchmark_name in benchmark_names:
        benchmark_result = results.get(benchmark_name)
        if not isinstance(benchmark_result, Mapping):
            raise RuntimeError(
                f'Driving evaluation returned no result for benchmark {benchmark_name!r}'
            )
        summary = benchmark_result.get('summary')
        metrics = summary.get('metrics_mean') if isinstance(summary, Mapping) else None
        if not isinstance(metrics, Mapping) or not metrics:
            raise RuntimeError(
                f'Driving evaluation returned no metrics for benchmark {benchmark_name!r}'
            )
        compact[benchmark_name] = {
            'summary': {
                key: copy.deepcopy(value)
                for key, value in summary.items()
                if key in ('num_scenarios', 'num_episodes', 'metrics_mean')
            }
        }
    return compact


def run_student_driving_evaluation(
    config: Mapping[str, Any],
    *,
    student: nn.Module,
    resolved_teacher_config: Mapping[str, Any],
    monitor: Any,
    progress: Any,
    step: int,
    run_dir: PathLike,
) -> Mapping[str, Any]:
    """Run one isolated native simulator evaluation for the live student."""
    driving_config = validate_driving_evaluation_config(config)
    if not driving_config.get('enabled', False):
        raise ValueError('run_student_driving_evaluation requires driving_evaluation.enabled=true')
    if not isinstance(resolved_teacher_config, Mapping):
        raise TypeError('resolved_teacher_config must be a mapping')
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError('evaluation step must be a non-negative integer')

    run_dir = Path(run_dir)
    evaluation_config = copy.deepcopy(dict(driving_config))
    evaluation_config.pop('enabled', None)
    evaluation_config.pop('interval_steps', None)
    evaluation_config['puffer_args'] = copy.deepcopy(dict(resolved_teacher_config))
    evaluation_config['output_root'] = str(run_dir.parent)
    evaluation_config['run_id'] = run_dir.name
    evaluation_config['output_subdir'] = _evaluation_output_subdir(run_dir, evaluation_config, step)
    evaluation_config['device'] = config['training']['device']
    evaluation_config['amp'] = False
    evaluation_config['compile'] = False
    evaluation_config['progress'] = {
        'optimizer_step': int(progress.optimizer_step),
        'simulator_transitions': int(progress.simulator_transitions),
        'collection_round_idx': int(progress.collection_round_idx),
        'update_epoch_idx': int(progress.update_epoch_idx),
    }

    resolved_env = evaluation_config['puffer_args'].get('env', {})
    if not isinstance(resolved_env, Mapping):
        raise ValueError('resolved_teacher_config.env must be a mapping')
    configured_env_overrides = config.get('env_overrides', {})
    if not isinstance(configured_env_overrides, Mapping):
        raise ValueError('config.env_overrides must be a mapping')
    env_overrides = copy.deepcopy(dict(configured_env_overrides))
    explicit_env_overrides = driving_config.get('env_overrides')
    if explicit_env_overrides is not None:
        env_overrides.update(copy.deepcopy(dict(explicit_env_overrides)))
    if 'dt' in env_overrides and 'dt' in resolved_env and env_overrides['dt'] != resolved_env['dt']:
        raise ValueError(
            'driving_evaluation.env_overrides.dt must match the resolved training env.dt'
        )
    if 'action_type' in env_overrides and env_overrides['action_type'] != 'continuous':
        raise ValueError('driving evaluation requires env_overrides.action_type=continuous')
    if env_overrides.get('resample_replay_to_dt'):
        raise ValueError('driving evaluation requires env_overrides.resample_replay_to_dt=false')
    for key in ('scenario_length', 'resample_frequency'):
        env_overrides.pop(key, None)
    # Evaluation must use the exact training observation recipe and native dt.
    for key in ('action_type', 'dt', 'resample_replay_to_dt'):
        if key in resolved_env:
            env_overrides[key] = copy.deepcopy(resolved_env[key])
    env_overrides['action_type'] = 'continuous'
    if 'dt' in resolved_env:
        env_overrides['dt'] = copy.deepcopy(resolved_env['dt'])
    env_overrides['resample_replay_to_dt'] = False
    evaluation_config['env_overrides'] = env_overrides

    training_vec = config.get('vec')
    if isinstance(training_vec, Mapping):
        vec_overrides = copy.deepcopy(dict(training_vec))
    else:
        vec_overrides = {}
    explicit_vec_overrides = driving_config.get('vec_overrides', driving_config.get('vec'))
    if explicit_vec_overrides is not None:
        vec_overrides.update(copy.deepcopy(dict(explicit_vec_overrides)))
    if vec_overrides:
        evaluation_config['vec_overrides'] = vec_overrides
    teacher_eval_config = evaluation_config['puffer_args'].get('eval', {})
    if not isinstance(teacher_eval_config, Mapping):
        raise ValueError('resolved_teacher_config.eval must be a mapping')
    training_env_agents = env_overrides.get('num_agents', resolved_env.get('num_agents'))
    if training_env_agents is None:
        training_env_agents = teacher_eval_config.get('num_agents')
    if isinstance(training_env_agents, bool) or not isinstance(training_env_agents, int) or training_env_agents <= 0:
        raise ValueError('driving evaluation requires a positive training agent count')
    requested_agent_count = driving_config.get('num_agents', training_env_agents)
    if requested_agent_count != training_env_agents:
        training_env_agents = requested_agent_count
    max_agents_per_env = resolved_env.get('max_agents_per_env')
    worker_count = vec_overrides.get('num_envs', 1)
    if isinstance(max_agents_per_env, int) and max_agents_per_env > 0 and isinstance(worker_count, int) and worker_count > 0:
        training_env_agents = min(training_env_agents, max_agents_per_env * worker_count)
    evaluation_config['num_agents'] = training_env_agents

    eval_args = dict(teacher_eval_config)
    for key, value in (
        ('render_scenarios', False),
        ('render_filter', None),
        ('failure_replay_csv', None),
        ('capture_observations', False),
        ('max_rendered_failures', None),
    ):
        eval_args[key] = value
    evaluation_config['puffer_args']['eval'] = eval_args

    rng_state, module_flags = _capture_evaluation_state(student)
    try:
        results = evaluate_student(evaluation_config, student=student, monitor=monitor)
        if not isinstance(results, Mapping):
            raise RuntimeError('Driving evaluation returned a non-mapping result')
        benchmark_names = evaluation_config.get('benchmarks', ['carla'])
        if isinstance(benchmark_names, str):
            benchmark_names = [item.strip() for item in benchmark_names.split(',') if item.strip()]
        return _compact_driving_evaluation_results(results, benchmark_names)
    finally:
        _restore_evaluation_state(rng_state, module_flags)


def validate_resume_config(current, saved):
    """Protect stored minibatch cursors and the scientific recipe from drift."""
    fields = {
        'training': ('seed', 'batch_size', 'update_epochs', 'learning_rate', 'weight_decay', 'gradient_clip_norm', 'optimizer', 'run_id'),
        'collection': ('transitions_per_round', 'split_seeds', 'action_selection'),
    }
    for section, names in fields.items():
        for name in names:
            if current.get(section, {}).get(name) != saved.get(section, {}).get(name):
                raise ValueError(f'Cannot change {section}.{name} when resuming stored training state')
    for section in ('model', 'loss', 'ema', 'teacher_config', 'teacher_checkpoint_sha256'):
        if current.get(section) != saved.get(section):
            raise ValueError(f'Cannot change {section} when resuming stored training state')


def validate_dataset_identity(manifest, teacher, config, split):
    """Reject teacher/layout/split mismatches before using stored trajectories."""
    expected_hash = getattr(teacher, 'condition_b_checkpoint_sha256', None)
    if not expected_hash or manifest.get('teacher_checkpoint_sha256') != expected_hash:
        raise ValueError('Dataset teacher checkpoint hash does not match the frozen teacher')
    if manifest.get('split') != split or manifest.get('split_seed') != config['collection']['split_seeds'][split]:
        raise ValueError(f'Dataset does not belong to the configured {split} split')
    expected_layout = teacher.condition_b_observation_layout
    actual_layout = manifest.get('observation_layout', {})
    if any(actual_layout.get(key) != value for key, value in expected_layout.items()):
        raise ValueError('Dataset observation layout does not match the frozen teacher')
    recorded_config = manifest.get('effective_config', {})
    if recorded_config.get('teacher_config') != config.get('teacher_config'):
        raise ValueError('Dataset effective teacher/environment recipe does not match this run')
    recorded_selection = recorded_config.get('collection', {}).get('action_selection')
    if recorded_selection != config['collection'].get('action_selection'):
        raise ValueError('Dataset teacher action selection does not match this run')
    for attribute, key in (('action_table', 'normalized_controls'), ('action_table_physical', 'physical_controls')):
        if not hasattr(teacher, attribute):
            continue
        expected_table = getattr(teacher, attribute).detach().cpu().numpy()
        recorded_table = np.asarray(manifest.get('action_layout', {}).get(key))
        if recorded_table.shape != expected_table.shape or not np.array_equal(recorded_table, expected_table):
            raise ValueError(f'Dataset {key} action table does not match the teacher')


def train(config, validation_batches=None, *, teacher=None, student=None, env=None, monitor=None):
    """Collect with a frozen teacher, update over that collection, and repeat.

    CPU simulator workers remain alive between rounds. Held-out data is fixed;
    optimizer, student, target encoder, and monitoring identity span all rounds.
    """
    settings_for_dispatch = config.get('training', {}) if isinstance(config, Mapping) else {}
    if isinstance(settings_for_dispatch, Mapping) and settings_for_dispatch.get('distributed'):
        from .distributed_train import train_distributed

        return train_distributed(
            config,
            validation_batches=validation_batches,
            teacher=teacher,
            student=student,
            env=env,
            monitor=monitor,
        )
    launcher_world_size = os.environ.get('WORLD_SIZE')
    if launcher_world_size is not None:
        try:
            launcher_world_size_value = int(launcher_world_size)
        except ValueError as exc:
            raise ValueError(f'WORLD_SIZE must be an integer, got {launcher_world_size!r}') from exc
        if launcher_world_size_value > 1:
            raise RuntimeError(
                'WORLD_SIZE>1 requires training.distributed=true; refusing to run the legacy single-GPU trainer'
            )
    from copy import deepcopy
    import json
    import time
    import subprocess
    from torch.utils.data import DataLoader
    from .collect import collect_dataset
    from .dataset import TrajectoryDataset
    from .model import ConditionBModel
    from .monitoring import MetricProgress, WandbMonitor
    from .runtime import create_vecenv, prepare_runtime, resolve_path
    from .teacher import load_teacher, resolve_teacher_config

    config = deepcopy(dict(config))
    driving_evaluation = validate_driving_evaluation_config(config)
    settings = config['training']
    collection = config['collection']
    for name in ('batch_size', 'update_epochs', 'max_optimizer_steps', 'validation_interval_steps', 'checkpoint_interval_steps'):
        if type(settings.get(name)) is not int or settings[name] < 1:
            raise ValueError(f'training.{name} must be a positive integer')
    if settings['batch_size'] < 2:
        raise ValueError('training.batch_size must be at least two')
    for name in ('num_collections', 'transitions_per_round', 'max_transitions', 'max_disk_bytes'):
        if type(collection.get(name)) is not int or collection[name] < 1:
            raise ValueError(f'collection.{name} must be a positive integer')
    if collection['num_collections'] * collection['transitions_per_round'] > collection['max_transitions']:
        raise ValueError('Requested collections exceed collection.max_transitions')
    for name in ('amp', 'compile', 'distributed'):
        if settings.get(name):
            raise ValueError(f'training.{name} is not supported by this trainer yet')
    if settings.get('optimizer', 'AdamW') != 'AdamW':
        raise ValueError('Only the configured AdamW optimizer is supported')
    run_id = settings.get('run_id')
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError('training.run_id must be a nonempty directory name')
    run_dir = resolve_path(settings['output_root']) / run_id
    resume_path = settings.get('resume_checkpoint')
    if run_dir.exists() and any(run_dir.iterdir()) and not resume_path:
        raise FileExistsError(f'Run directory already contains data: {run_dir}')
    split_seeds = collection['split_seeds']
    if len(set(split_seeds[name] for name in ('train', 'validation', 'test'))) != 3:
        raise ValueError('Training, validation, and test must use distinct collection seeds')
    prepare_runtime(config)
    config['code_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=resolve_path('.'), text=True).strip()
    config['teacher_config'] = resolve_teacher_config(config)
    teacher = teacher if teacher is not None else load_teacher(config, resolved_config=config['teacher_config'], device=settings['device'])
    config['teacher_checkpoint_sha256'] = getattr(teacher, 'condition_b_checkpoint_sha256', None)
    student = student if student is not None else ConditionBModel(config, teacher=teacher)
    student.to(settings['device'])
    optimizer = torch.optim.AdamW((parameter for parameter in student.parameters() if parameter.requires_grad),
                                 lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    if resume_path:
        saved_config = torch.load(resolve_path(resume_path), map_location='cpu', weights_only=False).get('config', {})
        validate_resume_config(config, saved_config)
    restored = load_checkpoint(resolve_path(resume_path), student, optimizer=optimizer, map_location=settings['device']) if resume_path else {}
    step = restored.get('step', 0)
    state = dict(restored.get('collection_state', {}))
    transitions = state.get('simulator_transitions', 0)
    sampler = dict(restored.get('sampler_state', {}))
    first_round = sampler.get('collection_round_idx', 0)
    driving_enabled = bool(driving_evaluation.get('enabled', False))
    driving_record = state.get('driving_evaluation')
    if not isinstance(driving_record, Mapping):
        driving_record = {}
    last_driving_evaluation_step = driving_record.get('step')
    last_driving_evaluation_results = driving_record.get('results', {})
    owned_env, owned_monitor = env is None, monitor is None
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2))
    dataset_root = resolve_path(collection['output_root']) / (collection.get('dataset_id') or run_id)
    exit_code = 1
    last_validation = {}
    interval_totals = {}
    interval_windows = 0
    interval_started = time.monotonic()
    try:
        monitor = monitor if monitor is not None else WandbMonitor(config['wandb'], config, run_dir,
                                                                  checkpoint_state=restored.get('monitoring_state') or None)
        env = env if env is not None else create_vecenv(config['teacher_config'], config, 'train')
        validation_path = state.get('validation_manifest') or settings.get('validation_manifest')
        if validation_batches is None:
            if validation_path is None:
                validation_config = deepcopy(config)
                validation_config['collection'].update(split='validation', transitions_per_round=settings['validation_transitions'])
                validation_env = create_vecenv(config['teacher_config'], validation_config, 'validation')
                try:
                    manifest = collect_dataset(validation_config, dataset_root / 'validation', teacher=teacher,
                                               env=validation_env, collection_round_idx=0)
                finally:
                    validation_env.close()
                validation_path = manifest['manifest_path']
            validation_path = resolve_path(validation_path)
            validation_manifest = json.loads(validation_path.read_text())
            validate_dataset_identity(validation_manifest, teacher, config, 'validation')
            validation_data = TrajectoryDataset(validation_path.parent, split='validation', manifest=validation_manifest)
            validation_batches = DataLoader(validation_data, batch_size=settings['batch_size'], shuffle=False, num_workers=0)
            state['validation_manifest'] = str(validation_path)
        state['collector_restarted_on_resume'] = bool(resume_path)

        def run_periodic_driving_evaluation(progress: MetricProgress, current_step: int) -> Mapping[str, Any]:
            nonlocal last_driving_evaluation_step, last_driving_evaluation_results
            results = run_student_driving_evaluation(
                config,
                student=student,
                resolved_teacher_config=config['teacher_config'],
                monitor=monitor,
                progress=progress,
                step=current_step,
                run_dir=run_dir,
            )
            last_driving_evaluation_step = current_step
            last_driving_evaluation_results = copy.deepcopy(dict(results))
            state['driving_evaluation'] = {
                'step': current_step,
                'results': copy.deepcopy(dict(results)),
            }
            return results

        for round_idx in range(first_round, collection['num_collections']):
            started = time.monotonic()
            reuse_collection = round_idx == first_round and sampler.get('manifest_path')
            if reuse_collection:
                manifest_path = Path(sampler['manifest_path'])
                manifest = json.loads(manifest_path.read_text())
            else:
                training_collection_config = deepcopy(config)
                training_dataset_root = dataset_root / 'train'
                reserved_bytes = sum(path.stat().st_size for path in dataset_root.rglob('*')
                                     if path.is_file() and not path.is_relative_to(training_dataset_root))
                remaining_disk_bytes = collection['max_disk_bytes'] - reserved_bytes
                if remaining_disk_bytes <= 0:
                    raise RuntimeError('Held-out artifacts exhaust collection.max_disk_bytes')
                training_collection_config['collection']['max_disk_bytes'] = remaining_disk_bytes
                manifest = collect_dataset(training_collection_config, training_dataset_root, teacher=teacher,
                                           env=env, collection_round_idx=round_idx)
                manifest_path = Path(manifest['manifest_path'])
                transitions += manifest['collection_transition_count']
                monitor.log_metrics({'collection/seconds': time.monotonic() - started,
                                     'throughput/collection_transitions_per_second': manifest['collection_transition_count'] / max(time.monotonic() - started, 1e-9),
                                     'collection/valid_windows': manifest['valid_window_count'],
                                     'collection/rejected_transitions': manifest['stats']['rejected_transition_count']},
                                    progress=MetricProgress(step, transitions, round_idx, 0))
            validate_dataset_identity(manifest, teacher, config, 'train')
            data = TrajectoryDataset(manifest_path.parent, split='train', manifest=manifest)
            if len(data) < 2:
                raise ValueError('Collection contains fewer than two valid training windows')
            start_epoch = sampler.get('update_epoch_idx', 0) if reuse_collection else 0
            for epoch_idx in range(start_epoch, settings['update_epochs']):
                generator = torch.Generator().manual_seed(settings['seed'] + round_idx * settings['update_epochs'] + epoch_idx)
                indices = torch.randperm(len(data), generator=generator).tolist()
                batches = [indices[offset:offset + settings['batch_size']] for offset in range(0, len(indices), settings['batch_size'])]
                if len(batches) > 1 and len(batches[-1]) == 1:
                    batches[-2].extend(batches.pop())
                start_batch = sampler.get('next_batch_idx', 0) if reuse_collection and epoch_idx == start_epoch else 0
                for batch_idx in range(start_batch, len(batches)):
                    if step >= settings['max_optimizer_steps']:
                        break
                    samples = [data[index] for index in batches[batch_idx]]
                    batch = TrainingBatch(*(torch.stack(values) for values in zip(*samples)))
                    losses = train_step(student, batch, optimizer, config=config)
                    step += 1
                    progress = MetricProgress(step, transitions, round_idx, epoch_idx)
                    sampler = {'collection_round_idx': round_idx, 'update_epoch_idx': epoch_idx,
                               'next_batch_idx': batch_idx + 1, 'manifest_path': str(manifest_path)}
                    state['simulator_transitions'] = transitions
                    window_count = batch.observations.shape[0]
                    interval_windows += window_count
                    for name, value in zip(losses._fields, losses):
                        interval_totals[name] = interval_totals.get(name, 0.0) + float(value) * window_count
                    if step == 1 or step % config['wandb']['log_interval_steps'] == 0:
                        metrics = {f'train/loss_{name}': value / interval_windows for name, value in interval_totals.items()}
                        metrics['train/gradient_norm'] = student.last_gradient_norm
                        metrics['train/learning_rate'] = optimizer.param_groups[0]['lr']
                        metrics['throughput/training_windows_per_second'] = interval_windows / max(time.monotonic() - interval_started, 1e-9)
                        monitor.log_metrics(metrics, progress=progress)
                        interval_totals, interval_windows = {}, 0
                        interval_started = time.monotonic()
                    if step % config['wandb'].get('diagnostics_interval_steps', 100) == 0:
                        monitor.log_metrics(representation_metrics(student, batch), progress=progress)
                    if step % settings['validation_interval_steps'] == 0:
                        last_validation = validate(student, validation_batches, config=config)
                        monitor.log_metrics({f'validation/{key}': value for key, value in last_validation.items()}, progress=progress)
                    if driving_enabled and step % driving_evaluation['interval_steps'] == 0:
                        run_periodic_driving_evaluation(progress, step)
                    if step % settings['checkpoint_interval_steps'] == 0:
                        save_checkpoint(run_dir / 'checkpoint.pt', student, optimizer=optimizer, step=step, config=config,
                                        sampler_state=sampler, collection_state=state, monitoring_state=monitor.state_dict())
                if step >= settings['max_optimizer_steps']:
                    break
            if step >= settings['max_optimizer_steps']:
                break
            sampler = {'collection_round_idx': round_idx + 1, 'update_epoch_idx': 0, 'next_batch_idx': 0}
        last_validation = validate(student, validation_batches, config=config)
        progress = MetricProgress(step, transitions, sampler.get('collection_round_idx', 0), sampler.get('update_epoch_idx', 0))
        if interval_windows:
            monitor.log_metrics({f'train/loss_{name}': value / interval_windows for name, value in interval_totals.items()}, progress=progress)
        monitor.log_metrics({f'validation/{key}': value for key, value in last_validation.items()}, progress=progress)
        if driving_enabled and last_driving_evaluation_step != step:
            run_periodic_driving_evaluation(progress, step)
        checkpoint_path = run_dir / 'final_model.pt'
        save_checkpoint(checkpoint_path, student, optimizer=optimizer, step=step, config=config,
                        sampler_state=sampler, collection_state=state, monitoring_state=monitor.state_dict())
        result = {'checkpoint': str(checkpoint_path), 'optimizer_steps': step,
                  'simulator_transitions': transitions, 'validation': last_validation,
                  'driving_evaluation': copy.deepcopy(last_driving_evaluation_results),
                  'monitoring': dict(monitor.state_dict())}
        (run_dir / 'result.json').write_text(json.dumps(result, indent=2))
        exit_code = 0
        return result
    finally:
        if owned_env and env is not None:
            env.close()
        if owned_monitor and monitor is not None:
            monitor.finish(exit_code=exit_code)


def train_step(student, batch, optimizer, *, config=None) -> LossTerms:
    """Apply a finite gradient update, then update the frozen target exactly once."""
    settings = config or {}
    student.train()
    device = next(student.parameters()).device
    batch = TrainingBatch(*(tensor.to(device) for tensor in batch))
    if batch.observations.shape[0] < 2:
        raise ValueError('Training needs at least two windows for the variance penalty')
    optimizer.zero_grad(set_to_none=True)
    outputs = student(batch.observations, batch.executed_controls)
    losses = student.compute_losses(outputs, batch.teacher_logits, config=settings)
    if not all(torch.isfinite(value).all().item() for value in losses):
        raise FloatingPointError('Non-finite loss; optimizer and EMA were not updated')
    losses.total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        settings.get('training', {}).get('gradient_clip_norm', 1.0),
        error_if_nonfinite=True,
    )
    optimizer.step()
    student.update_target_encoder(tau=settings.get('ema', {}).get('tau', 0.99))
    student.last_gradient_norm = float(gradient_norm)
    return LossTerms(*(value.detach() for value in losses))


@torch.no_grad()
def representation_metrics(student, batch):
    """Bounded latent-collapse and policy diagnostics on the current minibatch."""
    device = next(student.parameters()).device
    observations = batch.observations[:128, 0].to(device)
    latents = student.encode_context(observations)
    logits = student.decode_chunk(latents)
    standard_deviations = latents.std(dim=0, correction=0)
    singular_values = torch.linalg.svdvals(latents - latents.mean(dim=0))
    mass = singular_values / singular_values.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(mass * mass.clamp_min(1e-12).log()).sum())
    if singular_values.sum() == 0:
        effective_rank = torch.zeros_like(effective_rank)
    student_log_probs = logits.log_softmax(-1)
    teacher_log_probs = batch.teacher_logits[:128].to(device).log_softmax(-1)
    return {
        'representation/latent_std_mean': float(standard_deviations.mean()),
        'representation/latent_std_min': float(standard_deviations.min()),
        'representation/latent_norm_mean': float(latents.norm(dim=-1).mean()),
        'representation/effective_rank': float(effective_rank),
        'representation/student_entropy': float(-(student_log_probs.exp() * student_log_probs).sum(-1).mean()),
        'representation/teacher_entropy': float(-(teacher_log_probs.exp() * teacher_log_probs).sum(-1).mean()),
    }


def validation_batches_with_variance_support(batches):
    """Merge a final singleton into its preceding batch without dropping data."""
    previous = None
    for batch in batches:
        if batch.observations.shape[0] == 0:
            raise ValueError('Validation contains an empty batch')
        if previous is None:
            previous = batch
            continue
        if batch.observations.shape[0] == 1 or previous.observations.shape[0] == 1:
            previous = TrainingBatch(*(torch.cat(values, dim=0) for values in zip(previous, batch)))
            continue
        yield previous
        previous = batch
    if previous is not None:
        if previous.observations.shape[0] < 2:
            raise ValueError('Validation needs at least two windows for the variance penalty')
        yield previous


@torch.no_grad()
def validate(student, batches, *, config=None) -> Mapping[str, float]:
    """Sample-weighted held-out losses and per-slot KL; preserve mode and RNG."""
    was_training = student.training
    device = next(student.parameters()).device
    totals = {}
    sample_count = 0
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    student.eval()
    try:
        for batch in validation_batches_with_variance_support(batches):
            batch = TrainingBatch(*(tensor.to(device) for tensor in batch))
            outputs = student(batch.observations, batch.executed_controls)
            losses = student.compute_losses(outputs, batch.teacher_logits, config=config)
            metrics = {f'loss_{name}': value for name, value in zip(losses._fields, losses)}
            teacher_log_probs = batch.teacher_logits.log_softmax(dim=-1)
            student_log_probs = outputs.chunk_logits.log_softmax(dim=-1)
            slot_kl = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(-1).mean(0)
            metrics.update({f'teacher_kl/slot_{slot}': value for slot, value in enumerate(slot_kl)})
            metrics['teacher_kl/mean'] = slot_kl.mean()
            metrics['latent_std'] = outputs.context_latents.std(dim=0, correction=0).mean()
            metrics['latent_norm'] = outputs.context_latents.norm(dim=-1).mean()
            count = batch.observations.shape[0]
            sample_count += count
            for name, value in metrics.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f'Non-finite validation metric: {name}')
                totals[name] = totals.get(name, 0.0) + float(value) * count
        if sample_count == 0:
            raise ValueError('Validation requires at least one held-out window')
        return {name: value / sample_count for name, value in totals.items()}
    finally:
        student.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def save_checkpoint(path, student, *, optimizer=None, step=0, config=None,
                    sampler_state=None, collection_state=None, monitoring_state=None):
    """Atomically save student-only training state, including EMA and replay cursor.

    Simulator memory is not serialized. Resuming a stored collection is exact;
    subsequent collections start new simulator episodes, recorded in loop state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'condition_b_v1', 'model_config': student.export_metadata(),
        'model_state': student.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'step': step, 'config': dict(config or {}),
        'sampler_state': dict(sampler_state or {}),
        'collection_state': dict(collection_state or {}),
        'monitoring_state': dict(monitoring_state or {}),
        'rng_state': {'torch': torch.get_rng_state(), 'python': random.getstate(),
                      'numpy': np.random.get_state(),
                      'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None},
    }
    descriptor, temporary_path = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_checkpoint(path, student, *, optimizer=None, map_location='cpu'):
    """Load a trusted local Condition B checkpoint; reject PPO checkpoint formats."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get('format') != 'condition_b_v1':
        raise ValueError('Expected a Condition B checkpoint, not teacher PPO state')
    if payload['model_config'] != student.export_metadata():
        raise ValueError('Checkpoint model metadata differs from the configured student')
    if optimizer is not None and payload['optimizer_state'] is None:
        raise ValueError('Checkpoint has no optimizer state to resume')
    expected = student.state_dict()
    incoming = payload['model_state']
    if incoming.keys() != expected.keys() or any(incoming[key].shape != expected[key].shape for key in expected):
        raise ValueError('Checkpoint parameter names or shapes do not match the student')
    student.load_state_dict(incoming, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer_state'])
    rng = payload['rng_state']
    torch.set_rng_state(rng['torch'].cpu())
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
    return payload


def main():
    """CLI shared by the normal and bounded toy launchers."""
    import argparse
    from datetime import datetime, timezone
    import json
    from .runtime import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='project/jepa_distill/config/condition_b.yaml')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--run-id')
    parser.add_argument('--wandb-disabled', action='store_true')
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.set)
    if arguments.run_id:
        config['training']['run_id'] = arguments.run_id
    distributed_launch = bool(config.get('training', {}).get('distributed'))
    if config['training']['run_id'] is None and not distributed_launch:
        config['training']['run_id'] = 'condition_b_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    if arguments.wandb_disabled:
        config['wandb']['enabled'] = False
    result = train(config)
    launcher_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if launcher_world_size == 1 or int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
