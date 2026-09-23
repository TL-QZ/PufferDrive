"""Bounded torchrun probe of the production DDP update, without simulator collection.

Reuses windows from a completed toy checkpoint's collection. Each candidate must
run in a fresh process so an OOM cannot contaminate the next measurement. The
frozen teacher stays on-device, as it does during normal training. Screening uses
two microbatches/update; --full-batch verifies the 128,000-window effective batch.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from project.jepa_distill.distributed_train import _PooledStreamingDataset, _training_update
from project.jepa_distill.model import ConditionBModel
from project.jepa_distill.runtime import load_config, prepare_runtime
from project.jepa_distill.teacher import load_teacher, resolve_teacher_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', help='Completed toy checkpoint supplying window paths')
    source.add_argument('--manifest', help='Existing collection manifest; read only')
    parser.add_argument('--microbatch-size', required=True, type=int)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--full-batch', action='store_true')
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--profile', action='store_true', help='Profile the final update; its timing includes profiler overhead')
    args = parser.parse_args()
    if args.microbatch_size < 2 or args.microbatch_size > 128000 or args.updates < 2:
        parser.error('microbatch must be in [2,128000]; updates must be at least two')
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    world_size = dist.get_world_size()
    config = load_config('project/jepa_distill/config/condition_b.yaml')
    config['training']['device'] = str(device)
    config['training']['microbatch_size'] = args.microbatch_size
    prepare_runtime(config)
    teacher_config = resolve_teacher_config(config)
    teacher = load_teacher(config, resolved_config=teacher_config, device=device)
    student = ConditionBModel(config, teacher=teacher).to(device)
    ddp = DistributedDataParallel(student, device_ids=[device.index], broadcast_buffers=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False) if args.checkpoint else None
    sampler = checkpoint['distributed_state']['rank_states'][rank]['sampler_state'] if checkpoint else {}
    manifest_paths = [args.manifest] if args.manifest else sampler.get('manifest_paths')
    if not manifest_paths:
        # Final checkpoints can point past the completed final collection.
        run_id = checkpoint['config']['training']['run_id']
        root = Path(checkpoint['config']['collection']['output_root']) / run_id
        manifest_paths = sorted(root.glob('rank_*/train/round_0000/manifest.json'))
        if not manifest_paths:
            manifest_paths = sorted(root.glob('rank_*/round_0000/manifest.json'))
    dataset = _PooledStreamingDataset(manifest_paths)
    effective_windows = 128000 if args.full_batch else min(2 * args.microbatch_size, 128000)
    indices = [(rank * effective_windows + index) % len(dataset) for index in range(effective_windows)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {'rank': rank, 'microbatch_size': args.microbatch_size,
              'effective_windows_per_rank': effective_windows,
              'manifest_paths': [str(path) for path in manifest_paths],
              'updates': [], 'status': 'running'}
    destination = output / f'rank_{rank}.json'
    try:
        for update_idx in range(args.updates):
            dist.barrier(device_ids=[device.index])
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            profiling = args.profile and update_idx == args.updates - 1
            profiler = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
            ]) if profiling else nullcontext()
            with profiler:
                losses, global_count, gradient_norm = _training_update(
                    ddp, student, dataset, indices, optimizer, config=config,
                    device=device, world_size=world_size)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if profiling:
                averages = profiler.key_averages()
                (output / f'profile_rank_{rank}.txt').write_text(
                    averages.table(sort_by='self_cuda_time_total', row_limit=35) + '\n' +
                    averages.table(sort_by='self_cpu_time_total', row_limit=35))
                profiler.export_chrome_trace(str(output / f'trace_rank_{rank}.json'))
            report['updates'].append({
                'update': update_idx, 'seconds': elapsed,
                'global_windows_per_second': global_count / elapsed,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                'device_used_bytes_after_update': (
                    torch.cuda.mem_get_info(device)[1] - torch.cuda.mem_get_info(device)[0]),
                'device_total_bytes': torch.cuda.get_device_properties(device).total_memory,
                'loss': float(losses.total), 'gradient_norm': gradient_norm})
            destination.write_text(json.dumps(report, indent=2))
        report['status'] = 'passed'
        destination.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='oom', error=str(error))
        destination.write_text(json.dumps(report, indent=2))
        raise
    finally:
        dataset.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
