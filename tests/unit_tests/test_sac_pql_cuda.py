"""Small synthetic PQL CUDA checks; never create a driving environment."""

from collections import defaultdict

import pytest
import torch

from pufferlib import pufferl, pufferl_sac
from pufferlib.sac_sequence_replay import SequenceReplay
from test_sac_nstep_collection import (
    _ConstructorVecEnv, _NoopUtilization, _TinyActor, _TinyCritic,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("compiled", [False, True])
def test_pinned_replay_cuda_update_with_eager_and_compiled_bf16(monkeypatch, tmp_path, compiled):
    monkeypatch.setattr(pufferl, "Utilization", _NoopUtilization)
    monkeypatch.setattr(pufferl.PuffeRL, "print_dashboard", lambda *_args, **_kwargs: None)
    if compiled and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 device unavailable")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    replay = SequenceReplay(4, (1,), num_agents=1, pin_memory=True)
    learner = pufferl_sac.PuffeSAC(
        dict(
            data_dir=str(tmp_path), action_count=2, total_timesteps=8,
            batch_size=4, minibatch_size=4, max_minibatch_size=2,
            update_epochs=1, checkpoint_interval=None, return_estimator="pql",
            pql_horizon=4, pql_lambda=0.95, device="cuda", compile=compiled,
            amp=compiled, precision="bfloat16" if compiled else "float32",
        ),
        _ConstructorVecEnv(), _TinyActor(), _TinyCritic(), replay, reset_env=False,
    )
    for step in range(4):
        replay.add_sequence(
            torch.tensor([0]), torch.tensor([[float(step)]]), torch.tensor([step % 2]),
            torch.tensor([0.1 * step]), torch.tensor([[float(step + 1)]]),
            torch.tensor([0.0 if step == 3 else 0.99]),
            torch.tensor([step == 3]), torch.tensor([False]),
        )
    assert replay.get_batch(torch.arange(4))[0].is_pinned()
    offsets = torch.tensor([0, 4])
    targets = learner._prepare_pql_targets(torch.arange(4), offsets)
    assert targets.device.type == "cuda"
    assert targets.dtype == torch.float32 and not targets.requires_grad
    assert torch.isfinite(targets).all()
    assert targets[-1].item() == pytest.approx(0.3)
    before = [parameter.detach().clone() for parameter in learner.uncompiled_critic.parameters()]
    losses = defaultdict(float)
    updates = learner._train_sac_transition(losses, lambda *_args, **_kwargs: None, 0)
    torch.cuda.synchronize()
    assert updates == learner.learner_updates == 1
    assert learner.sampled_transitions == 4
    assert any(not torch.equal(old, new) for old, new in zip(before, learner.uncompiled_critic.parameters()))
    assert all(torch.isfinite(parameter).all() for parameter in learner.uncompiled_critic.parameters())
    assert torch.isfinite(learner.log_alpha)
