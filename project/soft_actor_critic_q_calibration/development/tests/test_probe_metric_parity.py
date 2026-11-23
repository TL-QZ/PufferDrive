"""Evaluation telemetry must not change policy inputs or driving transitions."""

import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.development.probe_env import SceneBatch, create_template
from project.soft_actor_critic_q_calibration.development.tests.test_probe_env import MAP_PATH, _env_config


@pytest.mark.skipif(not MAP_PATH.is_file(), reason="CARLA test map is not available")
@pytest.mark.parametrize("scene_seed", [7301, 7302])
def test_eval_metric_instrumentation_preserves_transitions(scene_seed):
    config = _env_config()
    config["scenario_length"] = 32
    templates = []
    batches = []
    try:
        for enabled in (False, True):
            template = create_template({**config, "compute_eval_metrics": enabled}, seed=101)
            templates.append(template)
            batches.append(SceneBatch.from_template(
                template, template.template_map_path, template.template_agent_count,
                copies=1, scene_seed=scene_seed,
            ))
        np.testing.assert_array_equal(batches[0].observations, batches[1].observations)
        generator = np.random.default_rng(scene_seed)
        for _ in range(config["scenario_length"]):
            actions = generator.uniform(-1, 1, size=batches[0].actions.shape).astype(np.float32)
            results = [batch.step(actions) for batch in batches]
            for key in ("observations", "rewards", "terminals", "truncations"):
                np.testing.assert_array_equal(results[0][key], results[1][key], err_msg=key)
            for key in ("pose_xyz", "speed_mps", "removed", "stopped", "reached_goal"):
                np.testing.assert_array_equal(results[0]["metadata"][key], results[1]["metadata"][key], err_msg=key)
    finally:
        for batch in batches:
            batch.close()
        for template in templates:
            template.close()
