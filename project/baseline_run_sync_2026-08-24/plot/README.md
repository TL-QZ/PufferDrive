# Evaluation plots

Plot one completed baseline CARLA evaluation by town:

```bash
./project/baseline_run_sync_2026-08-24/plot/plot_carla_metrics.sh 0
```

Plot one fine-tuned seed across CARLA, nuPlan single, and nuPlan multi:

```bash
./project/baseline_run_sync_2026-08-24/plot/plot_finetuned_benchmarks.sh 0
```

After both original and fine-tuned final evaluations exist, compare them:

```bash
./project/baseline_run_sync_2026-08-24/plot/plot_finetune_comparison.sh 0
```

Figures go to `project/baseline_run_sync_2026-08-24/output/`, which is ignored
by the experiment-local `.gitignore`.
