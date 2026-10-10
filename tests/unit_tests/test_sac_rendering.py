import json
import struct
import zlib

import numpy as np
import torch

from pufferlib.ocean.evaluation_utils.sac_eval_replay import SACReplayCapture
from pufferlib.sac_viz import encode_interactive_replay, render_interactive_replay_zlib


def _minimal_replay():
    replay = {
        "env": {
            "obs_slots_lane_n": 0,
            "obs_slots_boundary_n": 0,
            "obs_slots_traffic_controls_n": 0,
            "obs_slots_partners_n": 0,
            "num_goals": 1,
            "reward_conditioning": False,
            "action_type": "continuous",
            "dynamics_model": "classic",
        },
        "agent_f32": np.zeros((1, 1, 15), dtype=np.float32),
        "agent_i32": np.zeros((1, 1, 5), dtype=np.int32),
        "metrics_f32": np.zeros((1, 1, 18), dtype=np.float32),
        "puffer_f32": np.zeros((1, 1, 15), dtype=np.float32),
        "traffic_i16": np.zeros((1, 1, 6), dtype=np.int16),
        "raw_action": np.zeros((1, 1, 2), dtype=np.float32),
        "clipped_action": np.zeros((1, 1, 2), dtype=np.float32),
        "entropy": np.full((1, 1), 0.5, dtype=np.float32),
        "policy_probs": np.full((1, 1, 3), 1.0 / 3.0, dtype=np.float32),
    }
    return {}, replay


def test_sac_capture_accepts_value_arg_without_recording_it(tmp_path):
    capture = SACReplayCapture(
        args={"env": {}, "eval": {}},
        policy=None,
        replay_output_dir=tmp_path,
        capture_observations=False,
        num_workers=1,
        agents_per_batch=1,
        capture_batch_steps=1,
        episode_id_offset=0,
    )
    capture.capture_frame(
        obs=None,
        policy_obs_tensor=np.zeros((1, 1), dtype=np.float32),
        raw_action=np.zeros((1, 2), dtype=np.float32),
        action=np.zeros((1, 2), dtype=np.float32),
        logits=torch.zeros((1, 3), dtype=torch.float32),
        value=np.array([99.0], dtype=np.float32),
        logprob=torch.zeros(1, dtype=torch.float32),
        entropy=torch.full((1,), 0.5, dtype=torch.float32),
    )

    assert "value" not in capture.policy_history
    assert "entropy" in capture.policy_history
    assert "policy_probs" in capture.policy_history


def test_sac_replay_encoding_and_html_omit_value_but_keep_policy_data(tmp_path):
    scenario, replay = _minimal_replay()
    compressed_payload = encode_interactive_replay(scenario, replay)
    payload = zlib.decompress(compressed_payload)
    header_length = struct.unpack_from("<I", payload)[0]
    header = json.loads(payload[4 : 4 + header_length])

    assert "value" not in header["chunks"]
    assert {"entropy", "raw_action", "clipped_action", "policy_probs"} <= set(header["chunks"])

    replay_path = tmp_path / "sample.replay.zlib"
    html_path = tmp_path / "sample.html"
    replay_path.write_bytes(compressed_payload)
    render_interactive_replay_zlib(replay_path, html_path)
    html = html_path.read_text(encoding="utf-8")

    assert "C.value" not in html
    assert '<span class="name">value</span>' not in html
    assert "C.entropy[s]" in html
    assert "C.raw_action[ab+i]" in html
    assert "C.policy_probs[pb+i]" in html
