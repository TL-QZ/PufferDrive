import base64
import re

import pytest

from project.soft_actor_critic_q_calibration.development import rollout_viewer


def _payload():
    probabilities = [0.0] * 12
    probabilities[5] = 0.589904
    probabilities[11] = 0.318229
    probabilities[0] = 1.0 - sum(probabilities)
    actions = [
        {
            "action_idx": action_index,
            "final_outcome": "End: scene ended · Probe at fault: no",
            "q_min": -0.125,
            "recorded_return": -2.5,
            "actual_critic_target": None if action_index == 0 else 0.375,
            "action_label": f"action {action_index}",
            "initial_probability": probabilities[action_index],
            "replay_bytes": f"branch-{action_index}".encode("ascii"),
        }
        for action_index in range(12)
    ]
    return {
        "probe": {"probe_id": "probe", "recorded_repeat_idx": 0},
        "initial_probabilities": probabilities,
        "actions": actions,
    }


def test_write_viewer_bundles_sac_replays_and_adds_standard_action_grid(tmp_path):
    output_path = rollout_viewer.write_viewer(_payload(), tmp_path / "probe.html")
    html = output_path.read_text(encoding="utf-8")

    replay_chunks = re.findall(
        r'<script type="application/octet-stream" class="payload-chunk" data-action="(\d+)">(.*?)</script>',
        html,
    )
    assert len(replay_chunks) == 12
    assert [int(action_index) for action_index, _ in replay_chunks] == list(range(12))
    assert [base64.b64decode(chunk) for _, chunk in replay_chunks] == [
        f"branch-{action_index}".encode("ascii") for action_index in range(12)
    ]

    assert 'id="forced-action-grid" class="heat"' in html
    assert "button[data-action-index]" not in html
    assert "data-action-index=\\\"' + actionIndex" in html
    assert "actionIndex === selectedForcedAction ? ' selected' : ''" in html
    assert "aria-pressed=\\\"' + (actionIndex === selectedForcedAction)" in html
    assert "jerk_long" in html and "jerk_lat" in html
    assert 'const ACTION_SCORES = [[-0.125,null,-2.5],[-0.125,0.375,-2.5]' in html
    assert "Q min / training target" in html
    assert "querySelectorAll('.payload-chunk[data-action=\\\"' + actionIndex + '\\\"]')" in html
    assert "initReplay(actionIndex, currentStep)" in html
    assert 'id="hud-global"' in html
    assert 'id="obs-container"' in html
    assert 'data-pol="v"' not in html


def test_write_viewer_rejects_incomplete_action_grid(tmp_path):
    payload = _payload()
    payload["actions"].pop()

    with pytest.raises(ValueError, match="12 forced actions"):
        rollout_viewer.write_viewer(payload, tmp_path / "probe.html")


def test_write_viewer_rejects_probability_mismatch(tmp_path):
    payload = _payload()
    payload["actions"][5]["initial_probability"] = 0.5

    with pytest.raises(ValueError, match="does not match initial_probabilities"):
        rollout_viewer.write_viewer(payload, tmp_path / "probe.html")
