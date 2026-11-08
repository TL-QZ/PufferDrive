# PPO / SAC runtime investigation

## Completion evidence

1. Full historical workloads and profiler limits: `historical_summary.md` and JSON.
2. Ten controlled cases, including a repeated stock baseline: `results/matched_summary.md`.
3. Three production-size two-GPU cases with trained model weights: `results/production_summary.md`.
4. Isolated candidates, exact parity evidence, and 23 passing tests including CUDA; two separate operator traces: `results/trace_summary.md`.
5. Independent review completed; findings, reproducible commands and limitations: `REPORT.md` and `README.md`. Production learner files unchanged.

## Acceptance criteria

1. Reconstruct the actual PPO and SAC workloads from saved configs and history, distinguishing received rows, valid rows, retained rows, replay exposures and optimizer updates.
2. Run bounded real Drive PPO and SAC training on the same GPU model with resolved configs and raw per-cycle timing artifacts. Separate startup/compilation from warmed measurements and report sample counts, memory and device/process provenance.
3. Measure controlled variants: PPO filtering, SAC replay storage/gathering, and stock profiling overhead. Compare both actual work per received row and learner time per trained row. Include production microbatch geometry and distributed verification where resources allow.
4. Implement isolated SAC optimization candidates that preserve targets, sampling, optimizer order and gradient accumulation; verify numerical parity before drawing speedup conclusions.
5. Write a reproducible report explaining measured costs, achievable speedup, uncertainty, and whether comparable PPO wall time requires algorithm changes. Do not extrapolate short-run speed to policy quality or promise a 4B completion time.

## Experiment boundaries

- All harnesses, configs, probes and artifacts live in this folder. Initial production code stays unchanged.
- User authorized bounded profiling runs. Do not start another 4B production job.
- Use idle RTX A6000 devices 4/5 (verified at investigation start); recheck availability before launching. Devices 0-3 were busy.
- Disable remote logging, benchmark evaluation, checkpoint writes and automatic resume in timed experiments; separately quantify their historical contribution.
- Preserve full Drive architecture, observation dimensions and simulator settings unless an override is explicitly recorded.
- Do not compare overlapping GPU benchmark runs. Other jobs on the host can still affect CPU/memory bandwidth; report this limitation.
- Stock timings are frequency-scaled estimates. Main measurements synchronize at collection/train boundaries; intrusive per-region profiling is separate.

## Planned stages

1. Source and historical artifact audit.
2. Smoke validation of instrumentation and real environment integration.
3. Sequential warmed baseline/ablation measurements, then production geometry confirmation.
4. Semantics-preserving optimization parity tests and measured comparisons.
5. Independent review and final evidence audit.
