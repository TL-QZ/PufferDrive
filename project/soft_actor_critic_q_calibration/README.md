# SAC Q calibration

[Third-run binned rollout records](runs/third_20261122/README.md) · [Second-run reference distance distributions](runs/second_20261122/README.md) · [Browse first-run PQL renders](runs/first_20261114/pql/renders/README.md) · [Browse first-run PQL plots](runs/first_20261114/pql/plots/README.md) · [TES run](runs/first_20261114/tes/README.md) · [Development](development/README.md)

```text
development/                 shared code, native binding, tests, configs
docs/                        experiment design and validation
BRAINSTORM.md                investigation notes
learnings/                   conceptual notes
runs/<run>/<model>/
  scripts/                   commands for this model
  records/                   collected data, logs, summaries, analysis tables
  renders/                   interactive HTML episode replays
  plots/                     images grouped by analysis, one README per folder
```

Start analysis in a model's `plots/`; use its `records/` for measurements and `scripts/` to reproduce figures. Start implementation in `development/`.

[Brainstorm](BRAINSTORM.md) · [Design](docs/EXPERIMENT.md) · [Plot index](docs/PLOTS.md)
