# Validation — 2026-10-20

## Completed checks

- Config parity: trial 97 and 98 snapshots match their round 4_1 source settings
  except the documented experiment overrides. Both launcher dry runs passed;
  the tests verify output configs and prevent training during dry runs.
- Final combined regression command: **37 passed, 8 subtests passed**. It includes
  the new behavior/provider/config tests and existing warm-start interface and
  coordinator tests:

  ```bash
  source .venv/bin/activate
  OMP_NUM_THREADS=1 python -m pytest -q \
    tests/unit_tests/test_sac_behavior_provider.py \
    tests/unit_tests/test_ppo_init_behavior_policy.py \
    tests/unit_tests/test_sac_ppo_warm_start.py \
    tests/unit_tests/test_sac_warm_start_interfaces.py \
    project/soft_actor_critic_PPO_init_behavior_policy/tests/test_configuration_and_launcher.py
  ```

- Native-hook regression checks separately reported **48 passed** across
  provider, learner, final-observation, startup, evaluation, and loading tests.
  Counts overlap the combined suite; do not sum them. An existing CPU compile
  test reported an NVML-unavailable warning.
- Native `WandbLogger` construction, logging, and finish passed with mocked W&B
  for both resolved trial configs. W&B is enabled and the new group is correct;
  these checks did not create or verify an online run.
- Shell syntax and `git diff --check` passed. Independent review found a config
  test that regenerated snapshots before comparison; this was fixed. The test
  now checks existing snapshots, and separate generation tests write only to a
  temporary directory. Snapshot hashes remained unchanged across testing.

## Actual CPU simulator smokes

These used the exact user-selected PPO checkpoint, the real Drive simulator,
normal SAC actor/critic/temperature updates, and the new entry point. They used
reduced SAC networks, replay capacity 32 per rank, serial vector environments,
eight agent slots per rank, CPU float32, no compilation, no W&B, and no scheduled
benchmark evaluation. Teacher architecture and weights remained the actual model.

| Run | Global schedule | Final received steps | SAC logical updates | Final PPO replay share |
| --- | --- | --- | --- | --- |
| Single process | 64 PPO + 64 decay + 128 SAC | 256 | 168 | 0 |
| Two CPU/Gloo ranks | 128 PPO + 128 decay + 256 SAC | 512 | 168 | 0 |

Both runs exited successfully and saved `final_model.pt`, `trainer_state.pt`,
resolved config, and teacher provenance. All 601 tensors found recursively in
each trainer checkpoint were finite. Logs showed all three stages, decreasing
PPO collection probability, and eventual removal of PPO transitions from replay.

Temporary evidence locations on this machine:

```
/tmp/sac_ppo_behavior_smoke_retry_or14yvax/
/tmp/sac_ppo_behavior_ddp_cuadf96g/
/tmp/sac_ppo_behavior_tests.log
```

An initial integration smoke exposed a mismatched transition-source callback
signature; the native call and provider now agree, and both successful smokes
ran after the correction. No existing experiment artifacts were deleted.

## Limits

The full two-GPU, BF16, compiled, 8,192,000-transition-per-rank runs have not been
launched or validated. These checks do not establish runtime memory headroom,
throughput, benchmark quality, PPO/SAC score parity, or the effectiveness of the
250M/250M/500M schedule. The configured teacher's historical score is not a result
of this experiment. Exact interrupted-run resume is unsupported.
