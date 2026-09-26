from __future__ import annotations

import pytest

from project.jepa_distill.observation_probe.evaluate_checkpoint import (
    _prepare_output_dir,
    evaluate_checkpoint,
    main,
)


def test_checkpoint_evaluation_requires_integer_window_and_render_limits(tmp_path):
    with pytest.raises(ValueError, match="max_windows must be a positive integer"):
        evaluate_checkpoint("unused.pt", output_dir=tmp_path / "out", max_windows=1.5)
    with pytest.raises(ValueError, match="render_samples must be a non-negative integer"):
        evaluate_checkpoint("unused.pt", output_dir=tmp_path / "out", render_samples=True)


def test_checkpoint_evaluation_rejects_nonempty_output_directory(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _prepare_output_dir(str(output_dir))


def test_checkpoint_evaluation_cli_lists_standalone_options(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--probe-checkpoint",
        "--split",
        "--device",
        "--output-dir",
        "--max-windows",
        "--render-samples",
    ):
        assert option in help_text


def test_online_saved_decoder_evaluates_after_training_data_removed(tmp_path, monkeypatch):
    import importlib
    import torch
    from project.jepa_distill.observation_probe.config import load_probe_config
    from project.jepa_distill.observation_probe.train import _make_identity
    from project.jepa_distill.observation_probe import model, online

    evaluator = importlib.import_module('project.jepa_distill.observation_probe.evaluate_checkpoint')
    config = load_probe_config()
    source_path = tmp_path / 'frozen.pt'
    source_path.write_bytes(b'frozen source identity')
    validation_path = tmp_path / 'validation.json'
    validation_path.write_text('{}')
    config['checkpoint'] = str(source_path)
    config['data'].update(mode='online', collection_root=str(tmp_path / 'deleted_training'),
                          validation_manifest=str(validation_path))
    decoder = torch.nn.Linear(2, 2)
    probe_path = tmp_path / 'probe.pt'
    torch.save({'format': 'observation_probe_v1',
                'identity': _make_identity(config, source_path, []),
                'decoder_state': decoder.state_dict(),
                'training_mean_observation': [0.25, 0.5]}, probe_path)

    class Heldout:
        closed = False

        def __len__(self):
            return 4

        def close(self):
            self.closed = True

    heldout = Heldout()
    monkeypatch.setattr(evaluator, 'ProbeCollectionDataset', lambda paths, **kwargs: heldout)
    monkeypatch.setattr(evaluator, '_validate_heldout_compatibility', lambda *args, **kwargs: 'matched')
    monkeypatch.setattr(online, 'inspect_online_source', lambda path: {
        'compatibility': {'observation_layout': {'observation_dim': 2}}})
    monkeypatch.setattr(evaluator, 'preflight_probe', lambda config: pytest.fail('must not inspect old training data'))
    frozen = torch.nn.Linear(2, 2)
    frozen.latent_dim = 2
    monkeypatch.setattr(model, 'load_frozen_jepa', lambda *args, **kwargs: frozen)
    monkeypatch.setattr(model, 'ObservationDecoder', lambda *args, **kwargs: torch.nn.Linear(2, 2))

    def evaluate(*args, config):
        assert config['_training_mean_observation'] == [0.25, 0.5]
        assert not (tmp_path / 'deleted_training').exists()
        return {'reconstruction_loss': 0.5}

    monkeypatch.setattr(evaluator, 'evaluate_probe', evaluate)
    result = evaluator.evaluate_checkpoint(str(probe_path), output_dir=tmp_path / 'evaluation')
    assert result['metrics']['reconstruction_loss'] == 0.5
    assert heldout.closed
    validation_path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='identity no longer matches'):
        evaluator.evaluate_checkpoint(str(probe_path), output_dir=tmp_path / 'tampered_evaluation')
