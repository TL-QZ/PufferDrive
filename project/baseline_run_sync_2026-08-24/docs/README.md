# Baseline experiment documentation

Start with the document that matches what you are doing:

1. [Step-by-step code-reading guide](nuplan_dt03_code_reading_guide.md) — follow
   one setting from YAML through Python and C, then see how stepping and export
   consume the resampled data. Use this while navigating the source yourself.
2. [nuPlan dt=0.3 workflow](nuplan_dt03_workflow.md) — run sizing, training,
   resume, evaluation, rendering, and verification commands.
3. [Baseline context and notes](contexts_and_notes.md) — understand the complete
   CARLA-to-nuPlan experiment, timing contract, controller matrix, and naming.

The implementation changes core loading code because replay resampling must
happen before agent selection. Experiment settings, launchers, sizing, and
documentation remain isolated under `project/baseline_run_sync_2026-08-24`.
