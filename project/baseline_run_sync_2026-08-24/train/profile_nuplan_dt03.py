"""Bounded two-GPU sizing using the production checkpoint, collector and PPO update.

No production run discovery, W&B, evaluation, or checkpoint writes. Each candidate
gets a fresh process group and exactly one warm-up plus two measured cycles.
"""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import torch
import yaml

from finetune_config import PROJECT, cli_values, finetune_overrides

REPO = PROJECT.parents[1]
SESSION_SECONDS = 45 * 60
CYCLES = 3
GPU_HEADROOM = 0.15
HOST_HEADROOM = 0.20
MIN_IMPROVEMENT = 1.05


class ProbeComplete(Exception):
    pass


def probe_worker(options):
    from pufferlib import pufferl as training

    output = options.output_dir
    rank = int(os.environ["RANK"])
    config = finetune_overrides()
    config.update({
        "env.num_agents": options.agents,
        "train.max_minibatch_size": options.microbatch,
        "train.evaluation_interval_epochs": None,
        "train.checkpoint_interval": 1000000,
        "train.data_dir": str(output / "run"),
        "train.resume_state_path": None,
        "load_model_path": str(output / "initial_checkpoint/final_model.pt"),
        "run_name": "dt03_sizing_probe", "wandb": False, "neptune": False, "tb": False,
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
                "transitions": self.global_step - self.step_start,
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


def sample_process_tree(process):
    rss_bytes = 0
    pss_bytes = 0
    try:
        processes = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
    except psutil.NoSuchProcess:
        processes = []
    for child in processes:
        try:
            memory = child.memory_full_info()
            rss_bytes += memory.rss
            pss_bytes += getattr(memory, "pss", memory.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {"process_tree_rss_bytes": rss_bytes, "process_tree_pss_bytes": pss_bytes,
            "host_available_bytes": memory.available, "host_total_bytes": memory.total,
            "swap_used_bytes": swap.used, "swap_out_bytes": swap.sout}


def qualifying(rows, host):
    if len(rows) != 2 or any(len(rank_rows) != CYCLES for rank_rows in rows):
        return False
    if host["min_available_fraction"] < HOST_HEADROOM or host["swap_growth_bytes"] > 0:
        return False
    for row in [cycle for rank_rows in rows for cycle in rank_rows]:
        if not (row["finite_losses"] and row["synchronized_updates"] and row["synchronized_retained_counts"]
                and row["optimizer_updates"] > 0 and row["retained_transitions"] > 0
                and row["truncation_count"] > 0 and row["all_truncations_terminal"]):
            return False
        if (row["cuda_peak_reserved_bytes"] / row["device_total_bytes"] > 1 - GPU_HEADROOM
                or row["device_free_bytes"] / row["device_total_bytes"] < GPU_HEADROOM):
            return False
    return True


def stop_candidate(process):
    """Own the whole probe group, including workers, on timeout or interruption."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=1)
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=2)


def run_candidate(options, agents, microbatch, deadline):
    output = options.output_dir / f"agents{agents}_micro{microbatch}"
    output.mkdir()
    checkpoint_dir = output / "initial_checkpoint"
    checkpoint_dir.mkdir()
    for name in ("final_model.pt", "config.yaml"):
        (checkpoint_dir / name).symlink_to((options.source_run / name).resolve())
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc-per-node=2", "--max_restarts=0", str(Path(__file__).resolve()),
               "--worker", "--source-run", str(options.source_run), "--output-dir", str(output),
               "--agents", str(agents), "--microbatch", str(microbatch)]
    environment = dict(os.environ, WANDB_MODE="disabled", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    baseline_swap = psutil.swap_memory()
    host = {"peak_process_tree_rss_bytes": 0, "peak_process_tree_pss_bytes": 0,
            "min_available_fraction": 1.0, "swap_growth_bytes": 0}
    status = "completed"
    with (output / "probe.log").open("w") as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            max_samples = SESSION_SECONDS + 1
            for _ in range(max_samples):
                sample = sample_process_tree(process)
                host["peak_process_tree_rss_bytes"] = max(host["peak_process_tree_rss_bytes"], sample["process_tree_rss_bytes"])
                host["peak_process_tree_pss_bytes"] = max(host["peak_process_tree_pss_bytes"], sample["process_tree_pss_bytes"])
                host["min_available_fraction"] = min(host["min_available_fraction"], sample["host_available_bytes"] / sample["host_total_bytes"])
                host["swap_growth_bytes"] = max(host["swap_growth_bytes"], sample["swap_used_bytes"] - baseline_swap.used,
                                                sample["swap_out_bytes"] - baseline_swap.sout)
                if process.poll() is not None:
                    break
                if time.monotonic() >= deadline or host["min_available_fraction"] < HOST_HEADROOM or host["swap_growth_bytes"] > 0:
                    status = "timeout" if time.monotonic() >= deadline else "host_memory_limit"
                    break
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        finally:
            stop_candidate(process)

    rows = [json.loads(path.read_text()) for path in (output / "rank0.json", output / "rank1.json") if path.exists()]
    qualifies = status == "completed" and process.returncode == 0 and qualifying(rows, host)
    throughput = 0
    if qualifies:
        transitions = sum(row["transitions"] for rank_rows in rows for row in rank_rows[1:])
        seconds = sum(max(rank_rows[cycle]["cycle_seconds"] for rank_rows in rows) for cycle in (1, 2))
        throughput = transitions / seconds
    result = {"agents": agents, "microbatch": microbatch, "status": status,
              "returncode": process.returncode, "qualifies": qualifies,
              "global_transitions_per_second": throughput, "host": host, "ranks": rows}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def memory_allows_larger(previous, agents):
    if not previous["ranks"]:
        return False
    factor = agents / previous["agents"]
    peak_rows = [row for rank in previous["ranks"] for row in rank]
    if any(row["cuda_peak_reserved_bytes"] * factor > (1 - GPU_HEADROOM) * row["device_total_bytes"] for row in peak_rows):
        return False
    host_growth = previous["host"]["peak_process_tree_pss_bytes"] * (factor - 1)
    return psutil.virtual_memory().available - host_growth >= HOST_HEADROOM * psutil.virtual_memory().total


def select_candidate(results):
    qualified = [candidate for candidate in results if candidate["qualifies"]]
    if not qualified:
        return None
    fastest = max(candidate["global_transitions_per_second"] for candidate in qualified)
    tied = [candidate for candidate in qualified
            if fastest / candidate["global_transitions_per_second"] < MIN_IMPROVEMENT]
    return min(tied, key=lambda candidate: candidate["agents"])


def sizing_session(options):
    started = time.monotonic()
    # Reserve ten seconds for process-group teardown and writing the final report.
    deadline = started + SESSION_SECONDS - 10
    options.output_dir.mkdir(parents=True, exist_ok=False)
    hardware = {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(),
                "source_run": str(options.source_run), "session_limit_seconds": SESSION_SECONDS}
    if not hardware["cuda_available"] or hardware["device_count"] != 2:
        hardware["status"] = "blocked: exactly two usable CUDA GPUs are required"
        (options.output_dir / "summary.json").write_text(json.dumps(hardware, indent=2) + "\n")
        print(hardware["status"])
        return 2
    hardware["devices"] = [torch.cuda.get_device_name(rank) for rank in (0, 1)]
    if any(torch.cuda.get_device_properties(rank).total_memory < 45 * 1024**3 for rank in (0, 1)):
        raise ValueError("Sizing requires two RTX A6000-class GPUs with at least 45 GiB each")
    for name in ("final_model.pt", "config.yaml"):
        if not (options.source_run / name).is_file():
            raise FileNotFoundError(options.source_run / name)
    source_config = yaml.safe_load((options.source_run / "config.yaml").read_text())
    if source_config.get("rnn_name") is not None:
        raise ValueError("Sizing requires the feed-forward CARLA checkpoint (rnn_name: null)")
    results = []
    microbatch = 32000
    for agents in (16, 32, 64):
        if time.monotonic() >= deadline or (results and not memory_allows_larger(results[-1], agents)):
            break
        result = run_candidate(options, agents, microbatch, deadline)
        results.append(result)
        # The 16k fallback is only for an actual 32k CUDA OOM, never a general failure.
        log = (options.output_dir / f"agents{agents}_micro{microbatch}/probe.log").read_text()
        if agents == 16 and not result["qualifies"] and "CUDA out of memory" in log and time.monotonic() < deadline:
            microbatch = 16000
            results.append(run_candidate(options, agents, microbatch, deadline))
    selected = select_candidate(results)
    if selected is not None and selected["microbatch"] == 32000 and time.monotonic() < deadline:
        trial = run_candidate(options, selected["agents"], 64000, deadline)
        results.append(trial)
        if trial["qualifies"] and trial["global_transitions_per_second"] >= MIN_IMPROVEMENT * selected["global_transitions_per_second"]:
            selected = trial
    hardware.update(status="qualified" if selected else "no qualifying candidate", results=results,
                    selected=selected, elapsed_seconds=time.monotonic() - started)
    (options.output_dir / "summary.json").write_text(json.dumps(hardware, indent=2) + "\n")
    if selected:
        resources = {"env.num_agents": selected["agents"], "train.max_minibatch_size": selected["microbatch"],
                     "train.evaluation_interval_epochs": 640 // selected["agents"],
                     "train.checkpoint_interval": 128 // selected["agents"]}
        (options.output_dir / "selected_resources.yaml").write_text(yaml.safe_dump(resources))
    return 0 if selected else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--agents", type=int, choices=(16, 32, 64), default=32, help=argparse.SUPPRESS)
    parser.add_argument("--microbatch", type=int, choices=(16000, 32000, 64000), default=32000, help=argparse.SUPPRESS)
    options = parser.parse_args()
    options.source_run = options.source_run.resolve()
    options.output_dir = options.output_dir.resolve()
    if options.worker:
        probe_worker(options)
    else:
        raise SystemExit(sizing_session(options))
