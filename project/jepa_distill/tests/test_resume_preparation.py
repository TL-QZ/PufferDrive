"""Read-only resume planning and explicitly requested historical cache migration."""
import copy
import json
import sys

import pytest
import torch

from project.jepa_distill.prepare_resume import main, plan_resume


def _saved_run(tmp_path):
    config = {
        'training': {'world_size': 2, 'distributed': True, 'run_id': 'resume_test'},
        'collection': {'output_root': str(tmp_path / 'datasets'), 'num_collections': 3},
        'teacher_checkpoint_sha256': 'teacher-hash',
    }
    rank_states = []
    for rank_idx in range(2):
        root = tmp_path / 'datasets' / 'resume_test' / f'rank_{rank_idx:03d}' / 'train'
        for round_idx in range(3):
            directory = root / f'round_{round_idx:04d}'
            directory.mkdir(parents=True)
            (directory / 'manifest.json').write_text(json.dumps({
                'complete': True, 'split': 'train', 'teacher_checkpoint_sha256': 'teacher-hash',
            }))
        rank_states.append({'rank': rank_idx, 'rng_state': {'torch': 'saved'}, 'sampler_state': {
            'collection_round_idx': 1, 'update_epoch_idx': 2, 'next_batch_idx': 3,
            'manifest_path': str(root / 'round_0001' / 'manifest.json'),
        }})
    validation_path = tmp_path / 'validation' / 'manifest.json'
    validation_path.parent.mkdir()
    validation_path.write_text('{}')
    payload = {'format': 'condition_b_v1', 'config': copy.deepcopy(config),
               'step': 12, 'optimizer_state': {'state': {}},
               'distributed_state': {'world_size': 2, 'rank_states': rank_states},
               'collection_state': {'validation_manifest': str(validation_path)}}
    return config, payload


def test_resume_plan_is_read_only_and_preserves_active_and_validation(tmp_path):
    config, payload = _saved_run(tmp_path)
    plan = plan_resume(config, payload)
    assert [entry['round_idx'] for entry in plan['cleanup']] == [0, 2, 0, 2]
    assert 'not referenced' in plan['cleanup'][1]['reason']
    assert len(list((tmp_path / 'datasets').rglob('manifest.json'))) == 6
    assert plan['optimizer_step'] == 12


def test_explicit_cleanup_preserves_checkpoint_active_round(tmp_path, monkeypatch):
    config, payload = _saved_run(tmp_path)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    checkpoint_path = tmp_path / 'checkpoint.pt'
    torch.save(payload, checkpoint_path)
    monkeypatch.setattr(sys, 'argv', ['prepare_resume', '--config', str(config_path),
                                    '--checkpoint', str(checkpoint_path), '--cleanup-cached'])
    main()
    remaining = list((tmp_path / 'datasets').rglob('manifest.json'))
    assert len(remaining) == 2
    assert all(path.parent.name == 'round_0001' for path in remaining)
    assert (tmp_path / 'validation' / 'manifest.json').exists()
    assert checkpoint_path.exists()


def test_resume_plan_rejects_changed_loss_and_missing_active_data(tmp_path):
    config, payload = _saved_run(tmp_path)
    changed = copy.deepcopy(config)
    changed['loss'] = {'jepa_weight': 0.0}
    with pytest.raises(ValueError, match='loss'):
        plan_resume(changed, payload)
    active = tmp_path / 'datasets/resume_test/rank_000/train/round_0001/manifest.json'
    active.unlink()
    with pytest.raises(FileNotFoundError, match='manifest'):
        plan_resume(config, payload)


def test_resume_plan_rejects_symlinks_before_any_removal(tmp_path):
    config, payload = _saved_run(tmp_path)
    old_directory = tmp_path / 'datasets/resume_test/rank_001/train/round_0000'
    (old_directory / 'external').symlink_to(tmp_path / 'validation', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        plan_resume(config, payload)
    assert (tmp_path / 'datasets/resume_test/rank_000/train/round_0000/manifest.json').exists()
