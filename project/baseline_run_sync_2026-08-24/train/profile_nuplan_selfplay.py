"""Bounded scratch-initialized self-play sizing; no production training is launched.

Each candidate owns a process group, with one warm-up and two measured cycles.
Temporary runs and compilation caches are removed; only the report is retained.
"""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import psutil
import torch
import yaml

from finetune_config import cli_values
from selfplay_config import REPO, selfplay_overrides, resource_overrides
from profile_nuplan_dt03 import sample_process_tree, qualifying

SESSION_SECONDS = 45 * 60
CANDIDATE_SECONDS = 8 * 60
CYCLES = 3
GPU_HEADROOM = 0.15
HOST_HEADROOM = 0.20


class ProbeComplete(Exception):
    pass


def probe_worker(options):
    from pufferlib import pufferl as training

    output = options.output_dir
    rank = int(os.environ["RANK"])
    config = selfplay_overrides()
    config.update(resource_overrides(options.agents, options.workers, options.microbatch))
    config.update({
        "env.num_agents": options.agents,
        "train.max_minibatch_size": options.microbatch,
        "train.evaluation_interval_epochs": None,
        "train.checkpoint_interval": 1000000,
        "train.data_dir": str(output / "run"),
        "train.resume_state_path": None,
        "load_model_path": None,
        "run_name": "selfplay_sizing_probe", "wandb": False, "neptune": False, "tb": False,
    })
    sys.argv = [sys.argv[0], *cli_values(config)]
    args = training.load_config("puffer_drive")

    class MeasuredPPO(training.PuffeRL):
        def __init__(self, *args, **kwargs):
            self.host_peak_rss_bytes = 0
            self.host_peak_pss_bytes = 0
            self.host_monitor_stop = threading.Event()
            self.host_monitor = threading.Thread(target=self.monitor_host, daemon=True)
            self.host_monitor.start()
            super().__init__(*args, **kwargs)
            if self.optimizer.state or self.epoch != 0 or self.global_step != 0:
                raise RuntimeError("Probe did not start with a fresh optimizer and counters")
            self.optimizer_updates = 0
            self.loss_sample_count = 0
            self.finite_losses = True
            self.cycle_rows = []
            print(f"Rank {rank}: fresh weights, empty optimizer, zero counters", flush=True)
            self.optimizer.register_step_post_hook(self.count_update)

        def monitor_host(self):
            process = SimpleNamespace(pid=os.getpid())
            for _ in range(SESSION_SECONDS):
                sample = sample_process_tree(process)
                self.host_peak_rss_bytes = max(self.host_peak_rss_bytes, sample["process_tree_rss_bytes"])
                self.host_peak_pss_bytes = max(self.host_peak_pss_bytes, sample["process_tree_pss_bytes"])
                if self.host_monitor_stop.wait(1):
                    break

        def count_update(self, optimizer, arguments, keywords):
            self.optimizer_updates += 1

        def _ppo_loss(self, observations, *args, **kwargs):
            result = super()._ppo_loss(observations, *args, **kwargs)
            self.loss_sample_count += len(observations)
            self.finite_losses &= all(math.isfinite(value) for value in result[3].values())
            return result

        def print_dashboard(self, *args, **kwargs):
            pass

        def save_checkpoint(self):
            raise RuntimeError("Checkpoint writes are disabled during sizing")

        def evaluate(self):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.cycle_start = time.monotonic()
            self.step_start = self.global_step
            self.updates_start = self.optimizer_updates
            self.samples_start = self.loss_sample_count
            self.finite_losses = True
            result = super().evaluate()
            torch.cuda.synchronize()
            self.collection_seconds = time.monotonic() - self.cycle_start
            self.collection_free_bytes = torch.cuda.mem_get_info()[0]
            return result

        def train(self):
            optimization_start = time.monotonic()
            result = super().train()
            torch.cuda.synchronize()
            optimization_seconds = time.monotonic() - optimization_start
            device_free, device_total = torch.cuda.mem_get_info()
            row = {
                "rank": rank, "cycle": self.epoch, "warmup": self.epoch == 1,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "device_free_bytes": min(device_free, self.collection_free_bytes),
                "device_total_bytes": device_total,
                "process_tree_peak_rss_bytes": self.host_peak_rss_bytes,
                "process_tree_peak_pss_bytes": self.host_peak_pss_bytes,
                "collection_seconds": self.collection_seconds,
                "optimization_seconds": optimization_seconds,
                "cycle_seconds": time.monotonic() - self.cycle_start,
                "transitions": int(self.global_step - self.step_start),
                "retained_transitions": (self.loss_sample_count - self.samples_start) // self.config["update_epochs"],
                "optimizer_updates": self.optimizer_updates - self.updates_start,
                "finite_losses": self.finite_losses,
                "truncation_count": int(self.truncations.count_nonzero().item()),
                "all_truncations_terminal": bool(torch.all(self.terminals[self.truncations.bool()] == 1).item()),
            }
            peer_rows = [None, None]
            torch.distributed.all_gather_object(peer_rows, row)
            row["synchronized_updates"] = len({peer["optimizer_updates"] for peer in peer_rows}) == 1
            row["synchronized_retained_counts"] = len({peer["retained_transitions"] for peer in peer_rows}) == 1
            self.cycle_rows.append(row)
            rank_path = output / f"rank{rank}.json"
            pending_path = rank_path.with_suffix(".tmp")
            pending_path.write_text(json.dumps(self.cycle_rows, indent=2) + "\n")
            pending_path.replace(rank_path)
            print(f"Rank {rank}: cycle {self.epoch}/{CYCLES} finished", flush=True)
            if self.epoch == CYCLES:
                self.host_monitor_stop.set()
                self.host_monitor.join(timeout=2)
                raise ProbeComplete()
            return result

    training.PuffeRL = MeasuredPPO
    try:
        training.train("puffer_drive", args=args)
    except ProbeComplete:
        pass


def stop_candidate(process):
    """Terminate the owned process group even if its torchrun leader has exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=2)
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def run_candidate(work_dir, agents, workers, microbatch, deadline):
    output = work_dir / f"agents{agents}_workers{workers}_micro{microbatch}"
    output.mkdir()
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc-per-node=2", "--max_restarts=0", str(Path(__file__).resolve()),
               "--worker", "--output-dir", str(output), "--agents", str(agents),
               "--workers", str(workers), "--microbatch", str(microbatch)]
    environment = dict(os.environ, WANDB_MODE="disabled", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                       TORCHINDUCTOR_CACHE_DIR=str(work_dir / "inductor"),
                       TRITON_CACHE_DIR=str(work_dir / "triton"))
    baseline_swap = psutil.swap_memory()
    host = {"peak_process_tree_rss_bytes": 0, "peak_process_tree_pss_bytes": 0,
            "min_available_fraction": 1.0, "swap_growth_bytes": 0}
    utilization = []
    measured_utilization = []
    status = "completed"
    started = time.monotonic()
    candidate_deadline = min(deadline, started + CANDIDATE_SECONDS)
    print(f"Probe agents={agents}, workers={workers}, microbatch={microbatch}", flush=True)
    with (output / "probe.log").open("w") as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(CANDIDATE_SECONDS + 1):
                sample = sample_process_tree(process)
                host["peak_process_tree_rss_bytes"] = max(host["peak_process_tree_rss_bytes"], sample["process_tree_rss_bytes"])
                host["peak_process_tree_pss_bytes"] = max(host["peak_process_tree_pss_bytes"], sample["process_tree_pss_bytes"])
                host["min_available_fraction"] = min(host["min_available_fraction"], sample["host_available_bytes"] / sample["host_total_bytes"])
                host["swap_growth_bytes"] = max(host["swap_growth_bytes"], sample["swap_used_bytes"] - baseline_swap.used,
                                                sample["swap_out_bytes"] - baseline_swap.sout)
                gpu = subprocess.run(["nvidia-smi", "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
                                      "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     text=True, capture_output=True, timeout=3, check=True)
                utilization.append([int(value) for value in gpu.stdout.split()])
                rank_paths = [output / f"rank{rank}.json" for rank in (0, 1)]
                if all(path.exists() for path in rank_paths):
                    completed_cycles = [len(json.loads(path.read_text())) for path in rank_paths]
                    if max(completed_cycles) < CYCLES:
                        measured_utilization.append(utilization[-1])
                if process.poll() is not None:
                    break
                if time.monotonic() >= candidate_deadline or host["min_available_fraction"] < HOST_HEADROOM or host["swap_growth_bytes"] > 0:
                    status = "timeout" if time.monotonic() >= candidate_deadline else "host_memory_limit"
                    break
                time.sleep(min(1, max(0, candidate_deadline - time.monotonic())))
        finally:
            stop_candidate(process)
    rows = [json.loads(path.read_text()) for path in (output / "rank0.json", output / "rank1.json") if path.exists()]
    qualifies = status == "completed" and process.returncode == 0 and qualifying(rows, host)
    throughput = 0
    if qualifies:
        transitions = sum(row["transitions"] for rank_rows in rows for row in rank_rows[1:])
        seconds = sum(max(rank_rows[cycle]["cycle_seconds"] for rank_rows in rows) for cycle in (1, 2))
        throughput = transitions / seconds
    log_text = (output / "probe.log").read_text()
    result = {"agents": agents, "workers": workers, "microbatch": microbatch, "status": status,
              "returncode": process.returncode, "qualifies": qualifies,
              "elapsed_seconds": time.monotonic() - started,
              "global_transitions_per_second": throughput, "host": host, "ranks": rows,
              "gpu_utilization_mean_percent_including_startup": [sum(row[rank] for row in utilization) / len(utilization) for rank in (0, 1)],
              "gpu_utilization_peak_percent": [max(row[rank] for row in utilization) for rank in (0, 1)],
              "gpu_utilization_measured_mean_percent": [sum(row[rank] for row in measured_utilization) / len(measured_utilization) for rank in (0, 1)] if measured_utilization else None,
              "gpu_utilization_measured_sample_count": len(measured_utilization),
              "cuda_oom": "CUDA out of memory" in log_text,
              "failure_log_tail": log_text[-8000:] if not qualifies else None}
    print(f"Candidate qualifies={qualifies}, global transitions/s={throughput:.0f}, elapsed={result['elapsed_seconds']:.0f}s", flush=True)
    return result


def select_candidate(results):
    qualified = [result for result in results if result["qualifies"]]
    if not qualified:
        return None
    fastest = max(result["global_transitions_per_second"] for result in qualified)
    tied = [result for result in qualified if result["global_transitions_per_second"] >= fastest / 1.05]
    return min(tied, key=lambda result: max(row["cuda_peak_reserved_bytes"] for rank in result["ranks"] for row in rank))


def memory_allows_larger(results, agents):
    """Estimate growth from two measured sizes, keeping 10% prediction slack."""
    previous = results[-1]
    if not previous["qualifies"]:
        return False
    factor = agents / previous["agents"]
    comparable = [result for result in results if result["qualifies"]
                  and result["workers"] == previous["workers"]
                  and result["microbatch"] == previous["microbatch"]]
    for rank in (0, 1):
        peak = max(row["cuda_peak_reserved_bytes"] for row in previous["ranks"][rank])
        prediction = peak * factor
        if len(comparable) >= 2:
            earlier = comparable[-2]
            earlier_peak = max(row["cuda_peak_reserved_bytes"] for row in earlier["ranks"][rank])
            growth_per_agent = max(0, peak - earlier_peak) / (previous["agents"] - earlier["agents"])
            prediction = 1.10 * (peak + growth_per_agent * (agents - previous["agents"]))
        if prediction > (1 - GPU_HEADROOM) * previous["ranks"][rank][0]["device_total_bytes"]:
            return False
    host_growth = previous["host"]["peak_process_tree_pss_bytes"] * (factor - 1)
    return psutil.virtual_memory().available - host_growth >= HOST_HEADROOM * psutil.virtual_memory().total


def sizing_session(options):
    started = time.monotonic()
    deadline = started + options.max_seconds - 10
    options.output_dir.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly two usable CUDA GPUs are required")
    report = {"devices": [torch.cuda.get_device_name(rank) for rank in (0, 1)],
              "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "session_limit_seconds": options.max_seconds, "results": [],
              "config": selfplay_overrides()}
    results = report["results"]
    # Keep candidate runs and compiler caches together so interruption also cleans up.
    with tempfile.TemporaryDirectory(prefix="selfplay_probe_") as temporary:
        work_dir = Path(temporary)
        try:
            microbatch = 32000
            for agents in (512, 1024, 2048, 3200):
                if time.monotonic() >= deadline or (results and not memory_allows_larger(results, agents)):
                    break
                result = run_candidate(work_dir, agents, 20, microbatch, deadline)
                results.append(result)
                if agents == 512 and result["cuda_oom"] and time.monotonic() < deadline:
                    microbatch = 16000
                    results.append(run_candidate(work_dir, agents, 20, microbatch, deadline))
                if not results[-1]["qualifies"]:
                    break
            selected = select_candidate(results)
            if selected and selected["microbatch"] == 32000 and time.monotonic() < deadline:
                results.append(run_candidate(work_dir, selected["agents"], 20, 64000, deadline))
            selected = select_candidate(results)
            if selected:
                total_agents = selected["agents"] * selected["workers"]
                for workers in (10, 40):
                    if time.monotonic() >= deadline:
                        break
                    results.append(run_candidate(work_dir, total_agents // workers, workers, selected["microbatch"], deadline))
        finally:
            selected = select_candidate(results)
            report.update(status="qualified" if selected else "no qualifying candidate", selected=selected,
                          elapsed_seconds=time.monotonic() - started)
            (options.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if selected:
        resources = resource_overrides(selected["agents"], selected["workers"], selected["microbatch"])
        (options.output_dir / "selected_resources.yaml").write_text(yaml.safe_dump(resources))
    return 0 if selected else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=SESSION_SECONDS,
                        help="Session wall-clock cap, at most 2700 seconds")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--agents", type=int, default=512, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, choices=(10, 20, 40), default=20, help=argparse.SUPPRESS)
    parser.add_argument("--microbatch", type=int, choices=(16000, 32000, 64000), default=32000, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if not 30 <= options.max_seconds <= SESSION_SECONDS:
        parser.error("--max-seconds must be between 30 and 2700")
    options.output_dir = options.output_dir.resolve()
    if options.worker:
        probe_worker(options)
    else:
        raise SystemExit(sizing_session(options))
