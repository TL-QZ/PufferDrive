"""Learner invariants independent of simulator and teacher checkpoint availability."""
import copy

import pytest
import torch
from torch import nn

from project.jepa_distill.dataset import TrainingBatch
from project.jepa_distill.model import LossTerms, ModelOutputs
from project.jepa_distill.train import load_checkpoint, save_checkpoint, train_step, validate


class SmallStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.online = nn.Linear(3, 3)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.decoder = nn.Linear(3, 8)
        self.ema_updates = 0

    def forward(self, observations, controls):
        context = self.online(observations[:, 0])
        return ModelOutputs(context, self.decoder(context).reshape(-1, 4, 2), context,
                            self.target(observations[:, -1]))

    def compute_losses(self, outputs, teacher_logits, config=None):
        distillation = -(teacher_logits.softmax(-1) * outputs.chunk_logits.log_softmax(-1)).sum(-1).mean()
        jepa = (outputs.predicted_endpoint - outputs.target_endpoint).square().mean()
        variance = torch.relu(1 - outputs.context_latents.std(0, correction=0)).mean()
        return LossTerms(distillation + jepa + .1 * variance, distillation, jepa, variance)

    @torch.no_grad()
    def update_target_encoder(self, tau=.99):
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.mul_(tau).add_(online, alpha=1-tau)
        self.ema_updates += 1

    def export_metadata(self):
        return {'test_model': True}


def make_batch():
    return TrainingBatch(torch.randn(8, 5, 3), torch.randn(8, 4, 2), torch.randn(8, 4, 2))


def test_update_then_ema_and_frozen_target():
    student = SmallStudent()
    previous = [parameter.clone() for parameter in student.target.parameters()]
    optimizer = torch.optim.AdamW(student.parameters(), lr=.01)
    losses = train_step(student, make_batch(), optimizer, config={'ema': {'tau': .5}})
    assert torch.isfinite(losses.total)
    assert student.ema_updates == 1
    for old, target, online in zip(previous, student.target.parameters(), student.online.parameters()):
        torch.testing.assert_close(target, .5 * old + .5 * online)
        assert target.grad is None


def test_nonfinite_loss_never_updates_parameters_or_ema():
    student = SmallStudent()
    previous = copy.deepcopy(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters())
    batch = make_batch()
    batch.observations.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        train_step(student, batch, optimizer)
    assert student.ema_updates == 0
    for key, value in previous.items():
        torch.testing.assert_close(student.state_dict()[key], value)
    assert len(optimizer.state) == 0


def test_validation_preserves_parameters_mode_rng():
    student = SmallStudent().train()
    batch = make_batch()
    previous = copy.deepcopy(student.state_dict())
    rng = torch.get_rng_state().clone()
    metrics = validate(student, [batch])
    assert 'teacher_kl/slot_3' in metrics
    assert student.training and student.ema_updates == 0
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in previous.items():
        torch.testing.assert_close(student.state_dict()[key], value)


def test_checkpoint_resume_reproduces_next_optimizer_update(tmp_path):
    student = SmallStudent()
    batch = make_batch()
    optimizer = torch.optim.AdamW(student.parameters(), lr=.01)
    train_step(student, batch, optimizer)
    checkpoint = tmp_path / 'student.pt'
    save_checkpoint(checkpoint, student, optimizer=optimizer, step=1)
    expected_random = torch.rand(4)
    train_step(student, batch, optimizer)
    expected = copy.deepcopy(student.state_dict())
    restored = SmallStudent()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
    state = load_checkpoint(checkpoint, restored, optimizer=restored_optimizer)
    assert state['step'] == 1
    torch.testing.assert_close(torch.rand(4), expected_random)
    train_step(restored, batch, restored_optimizer)
    for key, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_validation_merges_singleton_without_dropping_samples():
    from project.jepa_distill.train import validation_batches_with_variance_support
    batch = make_batch()
    singleton = TrainingBatch(*(tensor[:1] for tensor in batch))
    merged = list(validation_batches_with_variance_support([batch, singleton]))
    assert len(merged) == 1
    assert merged[0].observations.shape[0] == 9
    torch.testing.assert_close(merged[0].observations[-1], singleton.observations[0])
    with pytest.raises(ValueError, match='at least two'):
        validate(SmallStudent(), [singleton])


@pytest.mark.parametrize('name', ['seed', 'batch_size', 'update_epochs'])
def test_resume_rejects_settings_that_change_batch_cursor(name):
    from project.jepa_distill.train import validate_resume_config
    saved = {'training': {'seed': 0, 'batch_size': 8, 'update_epochs': 2, 'max_optimizer_steps': 5}}
    current = copy.deepcopy(saved)
    current['training'][name] += 1
    with pytest.raises(ValueError, match=name):
        validate_resume_config(current, saved)
    current = copy.deepcopy(saved)
    current['training']['max_optimizer_steps'] = 10
    validate_resume_config(current, saved)


def test_dataset_identity_rejects_teacher_layout_or_split_drift():
    from types import SimpleNamespace
    from project.jepa_distill.train import validate_dataset_identity
    teacher = SimpleNamespace(condition_b_checkpoint_sha256='teacher-hash', condition_b_observation_layout={'observation_dim': 3})
    config = {'collection': {'split_seeds': {'validation': 1}}}
    manifest = {'teacher_checkpoint_sha256': 'teacher-hash', 'observation_layout': {'observation_dim': 3},
                'split': 'validation', 'split_seed': 1}
    validate_dataset_identity(manifest, teacher, config, 'validation')
    for field, value in [('teacher_checkpoint_sha256', 'another-teacher'), ('split_seed', 0),
                         ('split', 'train'), ('observation_layout', {'observation_dim': 4})]:
        corrupted = dict(manifest, **{field: value})
        with pytest.raises(ValueError):
            validate_dataset_identity(corrupted, teacher, config, 'validation')


def test_dataset_identity_rejects_different_dynamics_or_teacher_selection():
    from types import SimpleNamespace
    from project.jepa_distill.train import validate_dataset_identity
    teacher = SimpleNamespace(condition_b_checkpoint_sha256='hash', condition_b_observation_layout={'observation_dim': 3})
    config = {'teacher_config': {'env': {'dt': .3}}, 'collection': {'split_seeds': {'validation': 1}, 'action_selection': 'sample'}}
    manifest = {'teacher_checkpoint_sha256': 'hash', 'split': 'validation', 'split_seed': 1,
                'observation_layout': {'observation_dim': 3}, 'effective_config': copy.deepcopy(config)}
    validate_dataset_identity(manifest, teacher, config, 'validation')
    manifest['effective_config']['teacher_config']['env']['dt'] = .1
    with pytest.raises(ValueError, match='recipe'):
        validate_dataset_identity(manifest, teacher, config, 'validation')
    manifest['effective_config'] = copy.deepcopy(config)
    manifest['effective_config']['collection']['action_selection'] = 'mean'
    with pytest.raises(ValueError, match='action selection'):
        validate_dataset_identity(manifest, teacher, config, 'validation')
