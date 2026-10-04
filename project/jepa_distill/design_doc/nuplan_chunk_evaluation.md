# nuPlan chunk evaluation and comparison

- Evaluate the same JEPA and no-warm-start checkpoints used for CARLA, each at k=1–4 on `nuplan_single` and `nuplan_multi`.
- Reuse the existing chunk adapter and baseline benchmark catalog. Both nuPlan modes use seed 42, 1,000 scenarios, 200 steps at dt=0.1, mean actions, 300 agent slots per worker, and 20 workers. Preserve replay controllers and disable replay resampling as specified by the baseline catalog.
- Preserve CARLA defaults. Add explicit benchmark selection to its shared evaluator/launcher, then a thin nuPlan launcher that runs single and multi sequentially with four concurrent k jobs. Run the two checkpoints sequentially to cap concurrency at four jobs.
- Output contract: `<root>/<benchmark>/run_k<k>/k<k>/eval/<benchmark>_act_k<k>/standalone/evaluation_summary.json`, with per-k protocol metadata. Existing output roots are rejected; dry runs create no output.
- Add a nuPlan plotting entry point under `plot/`, reusing the CARLA metric groups and figure drawing. For each benchmark, compare k=1 against the same seed-0 baseline and plot k=1–4 with a horizontal baseline. Save figures and source/metric reports under `analysis_output/nuplan_comparison/` after evaluation.

Validation: targeted configuration/launcher tests, CARLA regression tests, real-checkpoint CPU dry runs, and synthetic plotting fixtures. Production evaluation is left for the user to launch. The k schedule stays unchanged: k=1 replans every 0.1 seconds; k=2/3/4 replan every 0.6/0.9/1.2 seconds.

## Run from the repository root

Choose four available GPUs. These commands run both checkpoints sequentially, eight evaluations each (two benchmarks times four k values):

```bash
export CUDA_VISIBLE_DEVICES=2,3,4,5
bash project/jepa_distill/scripts/evaluate_nuplan_4gpu.sh \
  --checkpoint experiments/jepa_distill/runs/condition_b_jepa_distill/checkpoint.pt \
  --output-dir experiments/jepa_distill/nuplan_eval_jepa_20260927 && \
bash project/jepa_distill/scripts/evaluate_nuplan_4gpu.sh \
  --checkpoint experiments/jepa_distill/runs/condition_b_distill_no_warm_start/final_model.pt \
  --output-dir experiments/jepa_distill/nuplan_eval_no_warm_start_20260927
```

Add `--dry-run` to each invocation to inspect GPU/job assignments without starting evaluation. This shell dry run checks launch arguments; the Python evaluator's `--device cpu --dry-run --benchmark nuplan_single` additionally loads and validates the checkpoint. Choose new output roots when repeating a run.

After both commands finish:

```bash
source .venv/bin/activate
python project/jepa_distill/plot/plot_nuplan_comparison.py \
  --jepa-root experiments/jepa_distill/nuplan_eval_jepa_20260927 \
  --no-warm-start-root experiments/jepa_distill/nuplan_eval_no_warm_start_20260927
```

Per-k logs live at `<root>/<benchmark>/k<k>.log`. Plots are grouped by benchmark under `project/jepa_distill/analysis_output/nuplan_comparison/`. No simulator run is needed to generate plots once the complete evaluation files exist.

## Validation status

- Both real checkpoints passed CPU dry-run validation for all 16 model/benchmark/k plans. No evaluation directories were created.
- Materialized configurations matched 14 key settings from the saved baseline nuPlan configurations, including timing, replay controllers, map paths, and resampling.
- Existing evaluation/CARLA rendering tests passed; new tests cover nuPlan configuration, launcher behavior, and plotting inputs/outputs. Plot fixtures generated 40 PNG/PDF files under temporary directories and tested rejection before output creation.
- Full simulator evaluation has not been run. Real nuPlan comparison plots become available after the launch commands above complete.
