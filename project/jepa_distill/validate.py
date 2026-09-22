"""Evaluate a saved student against its fixed held-out teacher trajectories."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from .dataset import TrajectoryDataset
from .model import ConditionBModel
from .runtime import prepare_runtime, resolve_path
from .train import validate, validate_dataset_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', help='Defaults to the checkpoint held-out validation manifest')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', help='Optional JSON metric output path')
    parser.add_argument('--wandb-disabled', action='store_true', help='Write local validation metrics only')
    arguments = parser.parse_args()
    checkpoint = torch.load(resolve_path(arguments.checkpoint), map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'condition_b_v1':
        raise ValueError('Expected a Condition B checkpoint')
    config = checkpoint['config']
    config['training']['device'] = arguments.device
    prepare_runtime(config)
    student = ConditionBModel(checkpoint['model_config']).to(arguments.device)
    student.load_state_dict(checkpoint['model_state'], strict=True)
    manifest_path = arguments.manifest or checkpoint['collection_state'].get('validation_manifest')
    if not manifest_path:
        raise ValueError('No held-out manifest in checkpoint; provide --manifest')
    manifest_path = resolve_path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    identity = SimpleNamespace(condition_b_checkpoint_sha256=config['teacher_checkpoint_sha256'],
                               condition_b_observation_layout=student.observation_layout)
    validate_dataset_identity(manifest, identity, config, 'validation')
    dataset = TrajectoryDataset(manifest_path.parent, split='validation', manifest=manifest)
    batches = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
    metrics = validate(student, batches, config=config)
    report = {'checkpoint': str(resolve_path(arguments.checkpoint)), 'validation': dict(metrics), 'windows': len(dataset)}
    from .monitoring import MetricProgress, WandbMonitor
    monitoring_config = dict(config['wandb'])
    if arguments.wandb_disabled:
        monitoring_config.update(enabled=False, mode='disabled')
    checkpoint_path = resolve_path(arguments.checkpoint)
    monitor = WandbMonitor(monitoring_config, config, checkpoint_path.parent,
                           checkpoint_state=checkpoint.get('monitoring_state'))
    exit_code = 1
    try:
        sampler = checkpoint.get('sampler_state', {})
        progress = MetricProgress(checkpoint['step'], checkpoint['collection_state']['simulator_transitions'],
                                  sampler.get('collection_round_idx', 0), sampler.get('update_epoch_idx', 0))
        monitor.log_metrics({f'validation/{key}': value for key, value in metrics.items()}, progress=progress)
        report['monitoring'] = dict(monitor.state_dict())
        exit_code = 0
    finally:
        monitor.finish(exit_code=exit_code)
    print(json.dumps(report, indent=2))
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
