
## 2026-09-22 goal regen patch
- drive.h regen block: bounded 32-attempt retry before removing agent; c_reset map-source branch now calls generate_new_goals_from_map. Unpatched run kept as *_seed0_UNPATCHED (dnf 0.556->0.71, carla_fast 0.960, nuplan_multi 0.182, nuplan_single 0.121 at epoch 39).
