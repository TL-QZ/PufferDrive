# Q-calibration organization

- Shared Python code, private native binding, tests, and configs belong in `development/`; design and validation documents belong in `docs/`. Keep user brainstorm and learnings in their existing locations.
- Every model result uses `runs/<run>/<model>/{scripts,records,plots,renders}/`. Put runnable model-specific commands in `scripts/`; use shared implementation instead of copying it.
- Store collected probes/reference trajectories, manifests, logs, numerical summaries, and supporting plot CSV/JSON/notes in `records/` (derived plot artifacts in `records/analysis/<plot-family>/`). Preserve historical execution metadata unchanged.
- Store images in `plots/<plot-family>/`; allow only images and one `README.md` per plot directory. Include directory READMEs explaining contents and navigation.
- Use `development/run_layout.py` and `development/plot_paths.py` for paths, update links when moving files, and never generate paired-action differences by default.

- Interactive HTML episode replays belong in the model’s `renders/` directory alongside `plots/`.
