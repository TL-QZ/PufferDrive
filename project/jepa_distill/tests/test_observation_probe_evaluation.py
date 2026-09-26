from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from project.jepa_distill.observation_probe.contracts import ProbeBatch, WindowSource
from project.jepa_distill.observation_probe.model import ObservationDecoder
from project.jepa_distill.observation_probe.schema import ObservationSchema


_LAYOUT = {
    "observation_dim": 1292, "dtype": "float32", "ego_features": 10,
    "num_reward_coefs": 17, "goal_dim": 9, "context_dim": 26,
    "goal_features": 3, "partner_features": 9, "lane_features": 9,
    "boundary_features": 9, "traffic_control_features": 7,
    "obs_slots_partners_n": 16, "obs_slots_lane_kept": 70,
    "obs_slots_boundary_kept": 50, "obs_slots_traffic_controls_n": 4,
    "obs_valid_count_features": 4,
}
_ENVIRONMENT = {
    "reward_conditioning": True, "num_goals": 3,
    "obs_slots_partners_n": 16, "obs_slots_lane_n": 70,
    "obs_slots_boundary_n": 50, "obs_slots_traffic_controls_n": 4,
    "obs_norm_goal_offset_m": 200.0, "obs_norm_xy_offset_m": 200.0,
    "obs_norm_veh_width_m": 2.0, "obs_norm_veh_length_m": 5.0,
    "obs_norm_road_seg_length_m": 5.0,
}
_GROUP_WEIGHTS = dict.fromkeys(
    ("ego", "context", "partners", "lanes", "boundaries", "traffic_controls"), 1.0
)


def _observation(schema: ObservationSchema, ego_value: float) -> torch.Tensor:
    observation = torch.zeros(schema.observation_dim, dtype=torch.float32)
    observation[schema.offsets["ego"]] = ego_value
    observation[schema.offsets["context"]] = 0.05
    rows = {
        "partners": [0.2, 0.1, 0.0, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0],
        "lanes": [0.3, 0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0],
        "boundaries": [0.3, -0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0],
        "traffic_controls": [0.2, 0.2, 0.4, 0.2, 0.0, 1.0, 2.0],
    }
    for group, row in rows.items():
        observation[schema.offsets[group]].reshape(schema.capacities[group], -1)[0] = torch.tensor(row)
    for index, _group in enumerate(schema.count_order):
        observation[schema.offsets["counts"].start + index] = 1.0
    return observation


class _FrozenJepa(torch.nn.Module):
    latent_dim = 4
    observation_layout = _LAYOUT

    def encode_target(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, : self.latent_dim]

    def encode_context(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, : self.latent_dim]

    def predict_endpoint(self, context: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        return context + controls.mean(dim=(1, 2)).unsqueeze(1)

    def export_metadata(self) -> dict:
        return {"teacher_config": {"env": _ENVIRONMENT}}


def _batch(schema, current_ego: float, future_ego: float, count: int, first_source: int):
    current = torch.stack([_observation(schema, current_ego + i * 0.01) for i in range(count)])
    future = torch.stack([_observation(schema, future_ego + i * 0.01) for i in range(count)])
    controls = torch.zeros((count, 4, 2), dtype=torch.float32)
    sources = tuple(WindowSource("synthetic-manifest", first_source + i) for i in range(count))
    return ProbeBatch(current, future, controls, sources)


def test_online_mean_uses_saved_statistics_without_any_training_files(tmp_path):
    from project.jepa_distill.observation_probe.evaluate import _training_mean

    schema = ObservationSchema(_LAYOUT)
    expected = _observation(schema, 0.25)
    config = {'data': {'mode': 'online', 'collection_root': str(tmp_path / 'deleted')},
              '_training_mean_observation': expected.tolist()}
    torch.testing.assert_close(_training_mean(config, schema), expected)
    config.pop('_training_mean_observation')
    with pytest.raises(ValueError, match='saved training mean'):
        _training_mean(config, schema)
    config['_training_mean_observation'] = [float('nan')] * schema.observation_dim
    with pytest.raises(ValueError, match='finite observation-sized'):
        _training_mean(config, schema)


def test_evaluate_probe_reports_finite_paths_caps_windows_and_saves_render(tmp_path, monkeypatch):
    from PIL import Image
    from project.jepa_distill.observation_probe import evaluate as evaluation
    from project.jepa_distill.observation_probe.evaluate import evaluate_probe

    schema = ObservationSchema(_LAYOUT)
    mean = _observation(schema, 0.2)
    monkeypatch.setattr(evaluation, "_training_mean", lambda _config, _schema: mean.clone())
    torch.manual_seed(47)
    decoder = ObservationDecoder(4, _LAYOUT, hidden_sizes=(8,))
    decoder.train()
    render_dir = tmp_path / "renders"
    config = {
        "data": {"collection_root": "unused", "collection_rounds": [0], "source_ranks": [0]},
        "evaluation": {"batch_size": 2, "max_windows": 5, "render_samples": 1,
                       "training_mean_windows": 1},
        "matching": {"method": "permutation_aware", "costs": {
            "continuous": 1.0, "categorical": 1.0, "presence": 0.2}},
        "loss": {"group_weights": _GROUP_WEIGHTS},
        "rendering": {"presence_threshold": 0.5},
        "_render_dir": str(render_dir),
    }
    metrics = evaluate_probe(
        _FrozenJepa(), decoder,
        [_batch(schema, 0.1, 0.4, 3, 0), _batch(schema, 0.1, 0.4, 4, 3)],
        config=config,
    )

    for path in ("reconstruction", "prediction", "persistence", "training_mean"):
        assert f"probe/{path}/loss_total" in metrics
        assert f"probe/{path}/ego/mae" in metrics
        assert np.isfinite(metrics[f"probe/{path}/loss_total"])
        assert np.isfinite(metrics[f"probe/{path}/ego/mae"])
    assert metrics["probe/windows"] == 5.0
    assert metrics["probe/persistence/ego/mae"] == pytest.approx(0.3, abs=1e-6)
    assert metrics["probe/persistence/ego/mae/elements"] == 50.0
    assert "probe/reconstruction/traffic_controls.type/accuracy" in metrics
    assert all(np.isfinite(value) for value in metrics.values())
    assert decoder.training

    files = sorted(render_dir.glob("comparison_*.png"))
    assert [path.name for path in files] == ["comparison_000.png"]
    with Image.open(files[0]) as image:
        assert image.mode == "RGB"
        assert image.size == (2160, 720)


def test_render_comparison_returns_three_rgb_panels_and_rejects_mismatched_width():
    from project.jepa_distill.observation_probe.render import render_comparison

    observation = _observation(ObservationSchema(_LAYOUT), 0.4).numpy()
    image = render_comparison(
        observation, observation.copy(), observation.copy(), environment_config=_ENVIRONMENT
    )
    assert image.shape == (720, 2160, 3)
    assert image.dtype == np.uint8
    with pytest.raises(ValueError, match="same layout width"):
        render_comparison(
            observation, observation[:-1], observation, environment_config=_ENVIRONMENT
        )


_TINY_LAYOUT = {
    "observation_dim": 51, "ego_features": 10, "num_reward_coefs": 0, "goal_dim": 3,
    "partner_features": 9, "lane_features": 9, "boundary_features": 9,
    "traffic_control_features": 7, "obs_slots_partners_n": 1,
    "obs_slots_lane_kept": 1, "obs_slots_boundary_kept": 1,
    "obs_slots_traffic_controls_n": 1, "obs_valid_count_features": 4,
}


def _empty_traffic_ddp_worker(rank: int, world_size: int, rendezvous: str, output_dir: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        from torch.nn.parallel import DistributedDataParallel
        from project.jepa_distill.observation_probe.losses import match_objects, reconstruction_loss

        torch.set_num_threads(1)
        decoder = ObservationDecoder(4, _TINY_LAYOUT, hidden_sizes=(8,))
        ddp_decoder = DistributedDataParallel(decoder)
        optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01)
        observations = torch.zeros((2, 51), dtype=torch.float32)
        matching = {"continuous": 1.0, "categorical": 1.0, "presence": 0.2}
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            latents = torch.full((2, 4), float(rank + update + 1))
            prediction = ddp_decoder(latents)
            assignments = match_objects(
                prediction, observations, observation_layout=_TINY_LAYOUT,
                matching_config=matching,
            )
            loss = reconstruction_loss(
                prediction, observations, assignments, observation_layout=_TINY_LAYOUT,
                loss_config={"group_weights": _GROUP_WEIGHTS},
            )
            loss.total.backward()
            optimizer.step()
        gradients = {
            name: decoder.categorical_heads[name].weight.grad.detach().cpu().clone()
            for name in ("traffic_controls__type", "traffic_controls__state")
        }
        torch.save(gradients, Path(output_dir) / f"rank_{rank}.pt")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_gloo_ranks_keep_empty_traffic_categorical_heads_in_ddp_graph(tmp_path):
    output_dir = tmp_path / "empty_traffic_gradients"
    output_dir.mkdir()
    mp.spawn(
        _empty_traffic_ddp_worker,
        args=(2, str(tmp_path / "empty_traffic_rendezvous"), str(output_dir)),
        nprocs=2,
        join=True,
    )
    rank_zero = torch.load(output_dir / "rank_0.pt", map_location="cpu", weights_only=True)
    rank_one = torch.load(output_dir / "rank_1.pt", map_location="cpu", weights_only=True)
    for name in ("traffic_controls__type", "traffic_controls__state"):
        assert torch.isfinite(rank_zero[name]).all()
        assert torch.count_nonzero(rank_zero[name]) == 0
        torch.testing.assert_close(rank_zero[name], rank_one[name])
