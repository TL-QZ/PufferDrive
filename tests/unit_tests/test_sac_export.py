import torch
import yaml

from pufferlib import pufferl, pufferl_sac

ACTOR_ENV_KEYS = {
    "dynamics_model",
    "num_goals",
    "reward_conditioning",
    "obs_slots_partners_n",
    "obs_slots_lane_n",
    "obs_dropout_lane",
    "obs_slots_boundary_n",
    "obs_dropout_boundary",
    "obs_slots_traffic_controls_n",
}


def _policy_config():
    return {
        "ego_input_size": 8,
        "partner_input_size": 8,
        "lane_input_size": 8,
        "boundary_input_size": 8,
        "traffic_control_input_size": 8,
        "context_input_size": 8,
        "backbone_hidden_size": 16,
        "backbone_num_layers": 1,
        "actor_hidden_size": 12,
        "actor_num_layers": 1,
        "critic_hidden_size": 12,
        "critic_num_layers": 1,
        "encoder_activation": "relu",
        "encoder_layer_norm": False,
        "backbone_activation": "gelu",
        "backbone_layer_norm": False,
        "shared_network": False,
        "mask_padded_features": True,
        "action_type": "discrete",
        "actor_head_layer_norm": False,
        "critic_head_layer_norm": False,
    }


def _source_config(dynamics_model):
    return {
        "dynamics_model": dynamics_model,
        "num_goals": 2 if dynamics_model == "classic" else 4,
        "reward_conditioning": dynamics_model == "jerk",
        "obs_slots_partners_n": 3 if dynamics_model == "classic" else 2,
        "obs_slots_lane_n": 7 if dynamics_model == "classic" else 5,
        "obs_dropout_lane": 0.25 if dynamics_model == "classic" else 0.4,
        "obs_slots_boundary_n": 5 if dynamics_model == "classic" else 8,
        "obs_dropout_boundary": 0.2 if dynamics_model == "classic" else 0.125,
        "obs_slots_traffic_controls_n": 2 if dynamics_model == "classic" else 3,
        "obs_norm_goal_offset_m": 333.0 if dynamics_model == "classic" else 425.0,
        "map_dir": "/source/data/maps",
    }


def _observation_batch(actor):
    backbone = actor.actor_backbone
    observation_size = (
        backbone.ego_dim
        + backbone.context_dim
        + backbone.obs_slots_partners_n * backbone.partner_features_count
        + backbone.obs_slots_lane_kept * backbone.lane_features_count
        + backbone.obs_slots_boundary_kept * backbone.boundary_features_count
        + backbone.obs_slots_traffic_controls_n * backbone.traffic_control_features_count
        + backbone.obs_valid_count_features
    )
    observations = torch.randn(3, observation_size)
    traffic_control_start = (
        backbone.ego_dim
        + backbone.context_dim
        + backbone.obs_slots_partners_n * backbone.partner_features_count
        + backbone.obs_slots_lane_kept * backbone.lane_features_count
        + backbone.obs_slots_boundary_kept * backbone.boundary_features_count
    )
    traffic_control_observations = observations[
        :, traffic_control_start:traffic_control_start
        + backbone.obs_slots_traffic_controls_n * backbone.traffic_control_features_count
    ].view(-1, backbone.obs_slots_traffic_controls_n, backbone.traffic_control_features_count)
    traffic_control_observations[:, :, backbone.traffic_control_continuous_features:] = 0
    observations[:, -backbone.obs_valid_count_features:] = torch.tensor(
        [
            [backbone.obs_slots_lane_kept, backbone.obs_slots_boundary_kept,
             backbone.obs_slots_partners_n, backbone.obs_slots_traffic_controls_n],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    return observations


def _run_export(tmp_path, dynamics_model):
    torch.manual_seed(37)
    source_actor_config = {
        "artifact_format": pufferl_sac.ACTOR_EXPORT_FORMAT,
        "artifact_version": pufferl_sac.ACTOR_EXPORT_VERSION,
        "policy_name": "DriveSAC",
        "rnn_name": None,
        "policy": _policy_config(),
        "env": _source_config(dynamics_model),
        "rnn": {},
    }
    source_actor = pufferl_sac._build_portable_actor(source_actor_config).eval()
    observations = _observation_batch(source_actor)
    with torch.no_grad():
        expected_logits = source_actor(observations)
        expected_action_ids = expected_logits.argmax(dim=-1)
        expected_normalized_actions = source_actor.discrete_actions_to_continuous(expected_action_ids)
        expected_physical_actions = source_actor.action_table_physical[expected_action_ids]

    run_dir = tmp_path / f"{dynamics_model}_run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "final_model.pt"
    torch.save(source_actor.state_dict(), checkpoint_path)
    source_config = {
        "policy_name": "DriveSAC",
        "rnn_name": None,
        "policy": _policy_config(),
        "rnn": {},
        "env": _source_config(dynamics_model),
    }
    with (run_dir / "config.yaml").open("w") as config_file:
        yaml.safe_dump(source_config, config_file)

    export_dir = tmp_path / f"{dynamics_model}_portable"
    pufferl_sac.main([
        "export",
        "puffer_drive",
        f"load_model_path={checkpoint_path}",
        f"+export_path={export_dir}",
    ])
    restored_actor = pufferl_sac.load_export(export_dir)

    with torch.no_grad():
        actual_logits = restored_actor(observations)
        actual_action_ids = actual_logits.argmax(dim=-1)
        actual_normalized_actions = restored_actor.discrete_actions_to_continuous(actual_action_ids)
        actual_physical_actions = restored_actor.action_table_physical[actual_action_ids]
    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_action_ids, expected_action_ids)
    torch.testing.assert_close(actual_normalized_actions, expected_normalized_actions)
    torch.testing.assert_close(actual_physical_actions, expected_physical_actions)

    with (export_dir / "config.yaml").open() as config_file:
        portable_config = yaml.safe_load(config_file)
    assert "map_dir" not in portable_config["env"]
    assert ACTOR_ENV_KEYS <= set(portable_config["env"])
    assert portable_config["env"]["obs_norm_goal_offset_m"] == source_config["env"]["obs_norm_goal_offset_m"]
    actor_weights = torch.load(export_dir / "actor.pt", map_location="cpu", weights_only=True)
    assert actor_weights
    assert all(key.startswith(("actor_backbone.", "actor_head.")) for key in actor_weights)
    assert not any("critic" in key or "optimizer" in key for key in actor_weights)

    # Existing SAC checkpoint loading can consume the plain actor state plus minimal config.
    reloaded_args, _ = pufferl.drive_benchmark.load_checkpoint_architecture({
        "load_model_path": str(export_dir / "actor.pt"),
        "policy": {},
        "rnn": {},
        "env": {},
        "train": {},
    })
    loader_actor = pufferl_sac._build_portable_actor(
        pufferl_sac._portable_actor_config(reloaded_args),
    )
    pufferl_sac._load_actor_weights(loader_actor, export_dir / "actor.pt", torch.device("cpu"), pufferl)
    with torch.no_grad():
        torch.testing.assert_close(loader_actor(observations), expected_logits)


def test_portable_export_rebuilds_classic_actor_without_source_paths(tmp_path):
    _run_export(tmp_path, "classic")


def test_portable_export_rebuilds_jerk_actor_without_source_paths(tmp_path):
    _run_export(tmp_path, "jerk")
