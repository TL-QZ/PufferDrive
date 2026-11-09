"""Freeze final checkpoints and the common CARLA protocol without changing runs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
RUNS = [
    ('tes', 'round2_tes_sac_20261029T044847.014369240_539543_trial_98', 'SAC', 'project/soft_actor_critic_target_entropy_scheduling/runs/round2_tes_sac_20261029T044847.014369240_539543/trial_98/learner', 0),
    ('nstep4', 'sac108_grid_20261031T194034Z658157854_trial_98', 'SAC', 'project/soft_actor_critic_multi_step_return/runs/round2_nstep4/sac108_grid_20261031T194034Z658157854/trial_98/learner', 1),
    ('nstep2', 'sac108_grid_20261102T220540Z788100439_trial_98', 'SAC', 'project/run_recovery/2026-11-05_seqamgpu01/nstep2_tes/learner', 4),
    ('pql_round4_1', 'sac108_grid_20261104T051212Z257259744_trial_98', 'SAC', 'project/run_recovery/2026-11-05_seqamgpu01/pql_round4_1/learner', 5),
    ('pql_official1', 'run_98_20261104T225607Z355122223_trial_98', 'SAC', 'project/soft_actor_critic_multi_step_return_PQL/runs/PQL_official_round1/run_98_20261104T225607Z355122223/trial_98/learner', 6),
    ('ppo', 'baseline_ppo_4b_carla_2026-11-06_13-58-32_seed0', 'PPO', 'experiments/baseline_ppo_4b_carla/baseline_ppo_4b_carla_2026-11-06_13-58-32_seed0', 7),
]
EXCLUDED = ['sac1_grid_20261104T230325Z530796082_trial_01', 'run_98_20261104T232616Z717021155_trial_98']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = OUTPUT / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'Already frozen: {manifest_path}')
    catalog_source = ROOT / 'project/baseline_ppo_4b_carla/override_config/evaluation_benchmarks.yaml'
    catalog = OUTPUT / 'evaluation_benchmarks.yaml'
    shutil.copy2(catalog_source, catalog)
    jobs = []
    for case, run_id, algorithm, relative_source, gpu in RUNS:
        source = ROOT / relative_source
        destination = OUTPUT / 'checkpoints' / case
        destination.mkdir(parents=True, exist_ok=False)
        for filename in ('final_model.pt', 'config.yaml'):
            shutil.copy2(source / filename, destination / filename)
        jobs.append(dict(case=case, run_id=run_id, algorithm=algorithm, gpu=gpu,
                         source_checkpoint=str(source / 'final_model.pt'),
                         checkpoint=str(destination / 'final_model.pt'),
                         checkpoint_sha256=digest(destination / 'final_model.pt'),
                         config_sha256=digest(destination / 'config.yaml'),
                         catalog=str(catalog), selection='final_model',
                         recovery='run_recovery' in relative_source))
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), jobs=jobs,
                    excluded_runs=[dict(run_id=run_id, reason='User identified unfinished; do not evaluate') for run_id in EXCLUDED],
                    catalog_source=str(catalog_source), catalog_sha256=digest(catalog),
                    protocol='Common PPO CARLA catalog: 1000 scenarios, seed42, 6000 steps, dt0.1, finite route goals spaced30m; mean actions; no rendering',
                    checkpoint_selection='Final checkpoints, independent of evaluation score; recovered runs use final recovery checkpoint')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Frozen {len(jobs)} final checkpoints; excluded {len(EXCLUDED)} unfinished runs.')


if __name__ == '__main__':
    main()
