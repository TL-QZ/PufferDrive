# Private Drive diagnostic binding

This directory builds a private copy of PufferDrive's existing Drive binding
with one read-only native query, `vec_get_diagnostic`. The generator reads the
shared `binding.c`, `env_binding.h`, and Drive headers without modifying them.
It writes generated sources, compiler intermediates, a digest stamp, and the
extension only under this directory. `diagnostic.py` installs the private
module into `pufferlib.ocean.drive.drive` in the current Python process; create
Drive environments after calling `install_private_binding()`.

## Use

From the repository root, activate or invoke `.venv` and call the wrapper
before constructing the analysis environment:

```python
from project.soft_actor_critic_q_calibration.diagnostic import (
    get_diagnostics,
    install_private_binding,
)

install_private_binding()
env = Drive(...)
metadata = get_diagnostics(env)  # query before choosing/stepping the action
```

`get_diagnostics(env)` returns NumPy arrays with one row per observation slot,
in observation order. `scene_idx` plus `slot_idx` identifies the row within the
current vector batch; `agent_id`, `episode_seed`, and `timestep` provide native
identity and episode context. The export includes pre-action mask and active
flags; lane and goal indices/count; position and active-goal XYZ coordinates;
Euclidean goal distance; raw current-lane-to-goal-lane graph distance,
reachability and status; heading, speed, reached-goal count/metric; and reward
coefficients. `reward_coefficients` columns follow
`diagnostic.REWARD_COEFFICIENT_NAMES`.

`graph_distance_status` values are `0=no_active_goal`,
`1=missing_current_or_goal_lane`, `2=lane_graph_unavailable`,
`3=current_or_goal_lane_absent_from_graph`,
`4=unreachable_or_invalid_graph_entry`, `5=reachable`, and
`6=inactive_or_padded_slot`. The raw graph matrix entry is retained for status
4 (including `inf`, negative, or `NaN` values); only status 5 is marked
reachable. Padded slots beyond a scene's active-agent count retain their stable
row identity and timestep, use `agent_id=-1`, set active flags to zero, and use
NaN for unavailable geometry and coefficients. A stopped or removed native
agent still has its real ID and last native state, while its active flags show
that it is no longer active.

The query reads native state and does not call reset/step, consume RNG, or alter
observations, rewards, or dynamics. The private extension is loaded only in the
calling Python process. It does not replace or rebuild the production
`binding` module on disk.

## Build and verification

Build locally (the wrapper also builds on demand when its source digest
changes):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  project/soft_actor_critic_q_calibration/native/build_private_binding.py --force
```

Run the CPU-only smoke and parity checks:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m \
  project.soft_actor_critic_q_calibration.native.verify_private_binding
```

Verified on 2026-11-02 with map
`data/nuplan_val/nuplan__00019774-39f3-5005-b94b-70cc71a2e557.bin`:

- Build completed successfully. GCC emitted no source warnings; the linker
  reported only its LTO notice about serial compilation of three LTRANS jobs.
- The route-generation failure case returned one row for the one observation
  slot: `slot_idx=0`, `agent_id=-1`, `episode_active=0`, and status 6.
- A two-scene, two-slot vector returned exactly two rows ordered as
  `scene_idx=[0,1]`, `slot_idx=[0,0]`; any padded row used the inactive sentinel.
- With `scenario_length=2` as a smoke-test-only override, the second transition
  returned truncation true and terminal false, captured a valid final
  observation, and reset to timestep 0 with a new episode seed.
- Production and private bindings produced bitwise-equal initial observations
  and three action transitions (observations, rewards, terminals, and
  truncations). Private metadata returned one pre-action row at each step.
- These checks ran on CPU. No training or GPU rollout is performed by this
  verifier.
