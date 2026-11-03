"""Snapshot Round 5 local evidence; run with the repository venv activated."""
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

OUTPUT = Path(__file__).resolve().parent / 'evidence/round5'
REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pufferdrive-round5-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    paths = sorted((REPO / 'project/soft_actor_critic_multi_step_return/runs/round5_nstep2_tes_sac').glob('*/trial_01/metrics.jsonl'))
    if len(paths) != 1:
        raise ValueError(f'Expected one Round 5 trial, found {len(paths)}')
    metrics_path = paths[0]
    provenance = []

    def read_snapshot(path):
        raw = path.read_bytes()
        provenance.append({'path': str(path.relative_to(REPO)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
        return raw

    raw = read_snapshot(metrics_path)
    if not raw.endswith(b'\n'):
        raise ValueError('Incomplete metrics record: retry after writer finishes')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = [row for row in rows if 'losses/entropy' in row]
    config = yaml.safe_load(read_snapshot(metrics_path.parent / 'learner/config.yaml'))
    evaluations = []
    for path in sorted((metrics_path.parent / 'learner/eval/training/carla_fast').glob('*/evaluation_summary.json')):
        match = re.fullmatch(r'epoch_(\d+)_step_(\d+)', path.parent.name)
        if match is None:
            raise ValueError(path)
        if int(match[2]) > rows[-1]['agent_steps']:
            continue
        report = json.loads(read_snapshot(path))
        evaluations.append({'epoch': int(match[1]), 'agent_steps': int(match[2]), **report['metrics_mean']})
        read_snapshot(path.parent / 'resolved_benchmark.yaml')
    evaluations.sort(key=lambda row: row['epoch'])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, data in [('training', rows), ('evaluations', evaluations)]:
        with (OUTPUT / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in data))))
            writer.writeheader()
            writer.writerows(data)
    zero_rows = [row for row in rows if row['losses/entropy'] == 0]
    diagnostic_keys = sorted(key for key in set().union(*(row.keys() for row in rows)) if any(token in key for token in ['diagnostic', 'logit', 'grad_norm', 'action_gap', 'max_action_probability', 'near_deterministic', 'td_error']))
    summary = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_records': len(rows), 'evaluation_records': len(evaluations),
        'latest': rows[-1], 'first_zero': zero_rows[0] if zero_rows else None,
        'zero_entropy_records': len(zero_rows),
        'minimum_alpha': min(rows, key=lambda row: row['losses/alpha']),
        'peak_evaluation': max(evaluations, key=lambda row: row['puffer_score']),
        'latest_evaluation': evaluations[-1],
        'diagnostic_keys': diagnostic_keys,
        'diagnostics_config': config['train'].get('log_collapse_diagnostics', 'absent (default false)'),
        'collapse_window': [row for row in rows if 464 <= row['epoch'] <= 480],
        'post_collapse_eval_unique_vectors': len({tuple(row[key] for key in ['puffer_score', 'avg_speed_per_agent', 'offroad_rate', 'collision_rate', 'num_goals_reached']) for row in evaluations if row['epoch'] >= 470}),
    }
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUTPUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    steps = [row['agent_steps'] / 1e6 for row in rows]
    for axis, keys, title in [
        (axes[0, 0], ['losses/entropy', 'losses/target_entropy'], 'Replay entropy (nats)'),
        (axes[0, 1], ['losses/alpha'], 'Temperature (log scale)'),
        (axes[1, 0], ['losses/critic_loss'], 'Critic loss (log scale)'),
        (axes[1, 1], ['losses/q1', 'losses/q2', 'losses/target'], 'Replay-action Q and target means'),
    ]:
        for key in keys:
            axis.plot(steps, [row[key] for row in rows], label=key.split('/')[-1])
        axis.set_title(title)
        axis.legend()
    axes[0, 1].set_yscale('log')
    axes[1, 0].set_yscale('log')
    for key in ['puffer_score', 'offroad_rate', 'collision_rate']:
        axes[2, 0].plot([row['agent_steps'] / 1e6 for row in evaluations], [row[key] for row in evaluations], label=key)
    axes[2, 0].set_title('Saved carla_fast evaluations')
    axes[2, 0].legend()
    for key in ['comfort', 'goal', 'offroad', 'reverse', 'lane_align']:
        full_key = 'environment/reward_components/' + key
        available = [row for row in rows if full_key in row]
        axes[2, 1].plot([row['agent_steps']/1e6 for row in available], [row[full_key] for row in available], label=key)
    axes[2, 1].set_title('Training episode reward components (sparse logs)')
    axes[2, 1].legend()
    for axis in axes.flat:
        axis.axvline(638.752, color='red', linestyle='--', alpha=.5)
        axis.grid(alpha=.2)
        axis.set_xlabel('Received rows (millions)')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'collapse.png', dpi=160)
    plt.close(fig)
    print(json.dumps({key: summary[key] for key in ['training_records', 'evaluation_records', 'zero_entropy_records', 'diagnostic_keys', 'post_collapse_eval_unique_vectors']}, indent=2))


if __name__ == '__main__':
    main()
