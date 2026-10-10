"""SAC evaluation rendering built around the supplied SAC actor."""

import copy
from datetime import datetime
import importlib
import os
import signal
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import pufferlib
import pufferlib.pytorch
import pufferlib.vector
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark
from pufferlib.ocean.evaluation_utils.sac_eval_replay import SACReplayCapture, _render_eval_replays


_previous_sigint_handler = signal.getsignal(signal.SIGINT)
try:
    from pufferlib import pufferl as ppo
finally:
    if signal.getsignal(signal.SIGINT) != _previous_sigint_handler:
        signal.signal(signal.SIGINT, _previous_sigint_handler)


def eval(
    env_name,
    args=None,
    policy=None,
    eval_output_dir=None,
    eval_output_subdir=None,
    use_training_config=False,
    benchmark_names=None,
):
    """Run benchmark evaluation and replay rendering with the supplied SAC actor."""
    if policy is None:
        raise pufferlib.APIUsageError("SAC evaluation requires the supplied SAC actor")
    cli_overrides = list(sys.argv[1:]) if args is None else []
    if any(override.split("=", 1)[0] == "env.num_agents" for override in cli_overrides):
        raise pufferlib.APIUsageError("Use eval.num_agents to configure evaluation agent count")
    if args is None:
        from pufferlib.pufferl_sac import load_config

        args = load_config(env_name)
    eval_config = args["eval"]
    render_scenarios = eval_config["render_scenarios"]
    render_filter = eval_config["render_filter"]
    failure_replay_csv = eval_config["failure_replay_csv"]
    capture_observations = eval_config["capture_observations"]

    benchmark_config_path = eval_config["benchmark_config"]
    selected_benchmarks = benchmark_names if benchmark_names is not None else eval_config["benchmarks"]
    eval_config["benchmarks"] = selected_benchmarks
    output_name = eval_config["output_name"]
    max_rendered_failures = eval_config["max_rendered_failures"]
    max_sdc_replay_workers = eval_config["max_sdc_replay_workers"]
    valid_action_selections = (
        pufferlib.pytorch.ACTION_SELECT_SAMPLE,
        pufferlib.pytorch.ACTION_SELECT_MODE,
        pufferlib.pytorch.ACTION_SELECT_MEAN,
    )
    if eval_config["action_selection"] not in valid_action_selections:
        raise pufferlib.APIUsageError(
            f"eval.action_selection='{eval_config['action_selection']}' must be one of {valid_action_selections}"
        )
    if render_scenarios and failure_replay_csv is not None:
        raise pufferlib.APIUsageError(
            "eval.render_scenarios requires a standard benchmark pass and cannot be combined "
            "with eval.failure_replay_csv"
        )
    if failure_replay_csv is not None and render_filter is None:
        raise pufferlib.APIUsageError("eval.failure_replay_csv requires eval.render_filter")

    report_to_wandb = bool(args["wandb"]) and not use_training_config
    environment_config, benchmarks = drive_benchmark.load_benchmark_config(
        benchmark_config_path, selected_benchmarks
    )
    if use_training_config:
        base_args = copy.deepcopy(args)
        environment_config["obs_dropout_lane"] = base_args["env"]["obs_dropout_lane"]
        environment_config["obs_dropout_boundary"] = base_args["env"]["obs_dropout_boundary"]
        checkpoint_config_path = None
    else:
        base_args, checkpoint_config_path = drive_benchmark.load_checkpoint_architecture(args)

    wandb_run_identity = (
        drive_benchmark.load_checkpoint_run_identity(checkpoint_config_path) if report_to_wandb else None
    )
    if eval_output_dir is None:
        run_dir = drive_benchmark.resolve_run_dir(base_args["load_model_path"])
        eval_output_dir = os.path.join(run_dir, eval_config["output_dir_name"])
    if eval_output_subdir is None:
        eval_output_subdir = datetime.now().strftime("%Y%m%d-%H%M%S")
    failure_replay_output_dir = None
    if failure_replay_csv is not None:
        failure_replay_csv = os.path.abspath(failure_replay_csv)
        if not os.path.isfile(failure_replay_csv):
            raise pufferlib.APIUsageError(
                f"eval.failure_replay_csv must reference an existing CSV file: {failure_replay_csv}"
            )
        failure_replay_output_dir = os.path.dirname(failure_replay_csv)
    benchmark_results = {}
    evaluation_policy_cache = {"policy": policy}
    cli_override_config = OmegaConf.from_dotlist(cli_overrides)
    for benchmark in benchmarks:
        run_args = drive_benchmark.build_benchmark_args(base_args, benchmark, environment_config)
        run_args = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(dict(run_args)), cli_override_config),
            resolve=True,
        )
        run_args["env"]["num_agents"] = run_args["eval"]["num_agents"]
        if run_args["env"]["simulation_mode"] == "replay" and run_args["env"]["control_mode"] == "control_sdc_only":
            run_args["vec"]["num_envs"] = min(run_args["vec"]["num_envs"], max_sdc_replay_workers)
        output_directory_name = benchmark["name"]
        if output_name is not None:
            output_directory_name = f"{output_directory_name}_{output_name}"
        if failure_replay_output_dir is not None:
            benchmark_output_dir = failure_replay_output_dir
            resolved_benchmark_output_dir = os.path.join(benchmark_output_dir, "failures")
            os.makedirs(resolved_benchmark_output_dir, exist_ok=True)
        else:
            benchmark_output_dir = os.path.join(eval_output_dir, output_directory_name)
            benchmark_output_dir = os.path.join(benchmark_output_dir, eval_output_subdir)
            os.makedirs(benchmark_output_dir)
            resolved_benchmark_output_dir = benchmark_output_dir
        drive_benchmark.write_resolved_benchmark_config(
            run_args,
            benchmark,
            benchmark_config_path,
            checkpoint_config_path,
            os.path.join(resolved_benchmark_output_dir, "resolved_benchmark.yaml"),
        )

        np.random.seed(run_args["train"]["seed"])
        if failure_replay_csv is not None:
            benchmark_results[benchmark["name"]] = _render_eval_failures(
                env_name,
                run_args,
                benchmark,
                failure_replay_csv,
                benchmark_output_dir,
                policy,
                capture_observations,
                max_rendered_failures,
                evaluation_policy_cache=evaluation_policy_cache,
            )
            continue

        num_scenarios = run_args["num_scenarios"]
        num_workers = min(run_args["vec"]["num_envs"], num_scenarios)
        worker_env_kwargs, total_steps = drive_benchmark._plan_benchmark_eval_workers(
            run_args,
            num_scenarios,
            num_workers,
            run_args["env"]["scenario_length"],
            capture_replay=render_scenarios,
        )
        print(f"Evaluation {benchmark['name']}: {num_scenarios} scenarios across {num_workers} workers")
        replay_output_dir = os.path.join(benchmark_output_dir, "replays") if render_scenarios else None
        summaries = _run_eval_rollout(
            run_args,
            env_name,
            worker_env_kwargs,
            total_steps,
            f"Evaluating {benchmark['name']}",
            num_scenarios,
            policy=policy,
            replay_output_dir=replay_output_dir,
            capture_observations=render_scenarios and capture_observations,
            evaluation_policy_cache=evaluation_policy_cache,
        )
        summary = drive_benchmark._write_eval_reports(summaries, benchmark_output_dir, num_scenarios)
        benchmark_results[benchmark["name"]] = {
            "episodes": summaries,
            "summary": summary,
        }

        if render_scenarios:
            _render_eval_replays(summaries, benchmark_output_dir)
        elif render_filter is not None:
            _render_eval_failures(
                env_name,
                run_args,
                benchmark,
                os.path.join(benchmark_output_dir, "episode_metrics.csv"),
                benchmark_output_dir,
                policy,
                capture_observations,
                max_rendered_failures,
                evaluation_policy_cache=evaluation_policy_cache,
            )

    if wandb_run_identity is not None:
        ppo.report_eval_to_wandb(args, benchmark_results, wandb_run_identity, eval_config["output_dir_name"])
    return benchmark_results


def _run_eval_rollout(
    args,
    env_name,
    worker_env_kwargs,
    total_steps,
    desc,
    expected_episodes,
    policy=None,
    recorded_agents_per_batch=None,
    replay_output_dir=None,
    capture_observations=False,
    episode_id_offset=0,
    evaluation_policy_cache=None,
):
    """Roll out the supplied SAC actor over workers and gather episode summaries."""
    num_workers = len(worker_env_kwargs)
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    vecenv = pufferlib.vector.make(
        [make_env] * num_workers,
        env_args=[[]] * num_workers,
        env_kwargs=worker_env_kwargs,
        backend="Multiprocessing",
        num_envs=num_workers,
        num_workers=num_workers,
        batch_size=num_workers,
        seed=args["vec"]["seed"],
    )
    scenario_progress = None
    try:
        agents_per_batch = vecenv.agents_per_batch
        inference_agents_per_batch = recorded_agents_per_batch or agents_per_batch
        if agents_per_batch > inference_agents_per_batch:
            raise pufferlib.APIUsageError(
                f"Replay environment batch has {agents_per_batch} agents, which exceeds the "
                f"CSV policy batch of {inference_agents_per_batch}. Reduce num_agents or the replay worker count."
            )
        if agents_per_batch < inference_agents_per_batch:
            print(
                f"Padding policy inference from {agents_per_batch} to "
                f"{inference_agents_per_batch} agents to preserve the recorded batch shape"
            )

        rollout_seed = args["train"]["seed"]
        torch.manual_seed(rollout_seed)
        if evaluation_policy_cache is None:
            evaluation_policy_cache = {"policy": policy}
        if evaluation_policy_cache.get("policy") is None:
            raise pufferlib.APIUsageError("SAC evaluation requires the supplied SAC actor")
        policy = evaluation_policy_cache["policy"]
        policy.eval()
        if "policy_forward_eval" not in evaluation_policy_cache:
            policy_forward_eval = policy.forward_eval
            eval_sample_logits = pufferlib.pytorch.sample_logits
            if args["train"]["compile"]:
                compile_kwargs = {
                    "mode": args["train"]["compile_mode"],
                    "fullgraph": args["train"]["compile_fullgraph"],
                }
                policy_forward_eval = torch.compile(policy_forward_eval, **compile_kwargs)
                eval_sample_logits = torch.compile(eval_sample_logits, **compile_kwargs)
            evaluation_policy_cache["policy_forward_eval"] = policy_forward_eval
            evaluation_policy_cache["sample_logits"] = eval_sample_logits
        policy_forward_eval = evaluation_policy_cache["policy_forward_eval"]
        eval_sample_logits = evaluation_policy_cache["sample_logits"]
        action_selection = args["eval"]["action_selection"]
        uncompiled_policy = ppo.base_policy(policy)
        env_continuous = isinstance(vecenv.single_action_space, pufferlib.spaces.Box)
        discrete_policy_on_continuous_env = env_continuous and not uncompiled_policy.is_continuous
        device = ppo.torch_device(args["train"]["device"])
        use_bfloat16 = args["train"]["amp"] and args["train"]["precision"] == "bfloat16" and ppo.is_cuda_device(device)
        if use_bfloat16 and not torch.cuda.is_bf16_supported():
            raise pufferlib.APIUsageError("bfloat16 evaluation requires CUDA BF16 support")
        eval_amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16)
        obs, _ = vecenv.reset(rollout_seed)
        ppo._require_finite_eval_batch(obs, "observations after eval reset", num_workers, worker_env_kwargs)
        padding_agent_count = inference_agents_per_batch - agents_per_batch
        policy_obs_tensor = None
        if padding_agent_count:
            policy_obs_tensor = torch.zeros(
                (inference_agents_per_batch, *obs.shape[1:]),
                dtype=torch.as_tensor(obs).dtype,
                device=device,
            )
        recurrent_state = None
        if args["train"].get("use_rnn", False):
            recurrent_state = {
                "lstm_h": torch.zeros(inference_agents_per_batch, policy.hidden_size, device=device),
                "lstm_c": torch.zeros(inference_agents_per_batch, policy.hidden_size, device=device),
            }

        capture_batch_steps = worker_env_kwargs[0]["resample_frequency"]
        replay_capture = None
        if replay_output_dir is not None:
            replay_capture = SACReplayCapture(
                args,
                policy,
                replay_output_dir,
                capture_observations,
                num_workers,
                agents_per_batch,
                capture_batch_steps,
                episode_id_offset,
            )

        episode_summaries = []
        scenario_progress = tqdm(total=expected_episodes, desc=desc, unit="scenario")
        for _ in range(total_steps):
            with torch.no_grad(), eval_amp_context:
                environment_obs_tensor = torch.as_tensor(obs, device=device)
                if padding_agent_count:
                    policy_obs_tensor[:agents_per_batch].copy_(environment_obs_tensor)
                else:
                    policy_obs_tensor = environment_obs_tensor
                if recurrent_state is None:
                    logits, value = policy_forward_eval(policy_obs_tensor)
                else:
                    logits, value = policy_forward_eval(policy_obs_tensor, recurrent_state)
                action, logprob, entropy, cont_action = eval_sample_logits(
                    logits,
                    action_selection=action_selection,
                    env_continuous=env_continuous,
                    policy=uncompiled_policy,
                )
                if discrete_policy_on_continuous_env:
                    raw_action = action[:agents_per_batch].cpu().numpy()
                    continuous_actions = cont_action.reshape(-1, *vecenv.single_action_space.shape)
                    action = continuous_actions[:agents_per_batch].float().cpu().numpy()
                else:
                    raw_action = action[:agents_per_batch].cpu().numpy().reshape(vecenv.action_space.shape)
                    action = raw_action
            if env_continuous:
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)
            ppo._require_finite_eval_batch(action, "policy actions", num_workers, worker_env_kwargs)

            if replay_capture is not None:
                replay_capture.capture_frame(
                    obs,
                    policy_obs_tensor,
                    raw_action,
                    action,
                    logits,
                    value,
                    logprob,
                    entropy,
                )

            obs, _, terminals, truncations, infos = vecenv.step(action)
            if recurrent_state is not None:
                finished_agent_mask = torch.as_tensor(
                    np.logical_or(terminals, truncations),
                    dtype=torch.bool,
                    device=device,
                ).reshape(agents_per_batch, 1)
                recurrent_state["lstm_h"][:agents_per_batch].masked_fill_(finished_agent_mask, 0)
                recurrent_state["lstm_c"][:agents_per_batch].masked_fill_(finished_agent_mask, 0)
            for worker_info in infos:
                worker_items = worker_info if isinstance(worker_info, list) else [worker_info]
                for item in worker_items:
                    if not isinstance(item, dict) or item.get("summary_type") != "evaluation_episode":
                        continue
                    if replay_capture is not None:
                        replay_capture.queue_replay(item, len(episode_summaries))
                    else:
                        item.pop("replay_environment_bundle", None)
                    item["agents_per_batch"] = inference_agents_per_batch
                    episode_summaries.append(item)
                    if len(episode_summaries) <= expected_episodes:
                        scenario_progress.update(1)

            if replay_capture is not None and replay_capture.pending_count:
                scenario_progress.set_postfix_str(f"writing {replay_capture.pending_count} replays")
                replay_capture.write_pending()
                scenario_progress.set_postfix_str("")

            if len(episode_summaries) >= expected_episodes:
                break
    finally:
        if scenario_progress is not None:
            scenario_progress.close()
        vecenv.close()
    if len(episode_summaries) != expected_episodes:
        print(
            f"WARNING: Evaluation expected {expected_episodes} episode summaries, "
            f"but received {len(episode_summaries)}. Writing the available results.",
            file=sys.stderr,
        )
    return episode_summaries


def _render_eval_failures(
    env_name,
    run_args,
    benchmark,
    metrics_path,
    benchmark_output_dir,
    policy,
    capture_observations,
    max_rendered_failures,
    evaluation_policy_cache=None,
):
    configured_render_filter = run_args["eval"]["render_filter"]
    selected_rows = drive_benchmark.select_render_rows(metrics_path, configured_render_filter)
    if max_rendered_failures is not None:
        selected_rows = selected_rows.head(max_rendered_failures).copy()
    failures_dir = os.path.join(benchmark_output_dir, "failures")
    os.makedirs(failures_dir, exist_ok=True)
    selected_path = os.path.join(failures_dir, "selected_failures.csv")
    selected_rows.to_csv(selected_path, index=False)
    if selected_rows.empty:
        print(f"No failures matched for benchmark {benchmark['name']}; wrote {selected_path}")
        return {"episodes": [], "summary": None}

    map_indices = drive_benchmark._resolve_map_indices(
        run_args["env"]["map_dir"],
        selected_rows["map_name"].tolist(),
    )
    seeds = pd.to_numeric(selected_rows["seed"], errors="raise").astype(np.int64).tolist()
    pairs = list(zip(map_indices, seeds))
    failure_args = copy.deepcopy(run_args)
    configured_worker_count = failure_args["vec"]["num_envs"]
    if configured_worker_count <= 0:
        raise pufferlib.APIUsageError("Failure rendering requires at least one worker")
    replay_wave_size = len(pairs)
    if capture_observations:
        observation_replay_wave_size = run_args["eval"]["observation_replay_wave_size"]
        if (
            isinstance(observation_replay_wave_size, bool)
            or not isinstance(observation_replay_wave_size, int)
            or observation_replay_wave_size <= 0
        ):
            raise pufferlib.APIUsageError(
                "eval.observation_replay_wave_size must be a positive integer when rendering observations"
            )
        replay_wave_size = min(
            len(pairs),
            configured_worker_count,
            observation_replay_wave_size,
        )
        replay_agent_capacity = failure_args["env"]["max_agents_per_env"]
        if replay_agent_capacity <= 0:
            raise pufferlib.APIUsageError("Failure rendering requires max_agents_per_env > 0")
        failure_args["env"]["num_agents"] = replay_agent_capacity
    replay_output_dir = os.path.join(failures_dir, "replays")
    os.makedirs(replay_output_dir, exist_ok=True)
    agents_per_batch_values = selected_rows["agents_per_batch"].unique()
    if len(agents_per_batch_values) != 1:
        raise pufferlib.APIUsageError("Benchmark failure rows must contain exactly one agents_per_batch value")
    recorded_agents_per_batch = int(agents_per_batch_values[0])
    summaries = []
    replay_wave_count = (len(pairs) + replay_wave_size - 1) // replay_wave_size
    for replay_wave_idx, replay_pair_start in enumerate(range(0, len(pairs), replay_wave_size)):
        replay_pairs = pairs[replay_pair_start : replay_pair_start + replay_wave_size]
        num_workers = min(configured_worker_count, len(replay_pairs))
        failure_args["vec"]["num_envs"] = num_workers
        worker_env_kwargs, total_steps = drive_benchmark._plan_failure_replay_workers(
            failure_args,
            replay_pairs,
            num_workers,
            failure_args["env"]["scenario_length"],
        )
        replay_desc = f"Rendering {benchmark['name']} failures"
        if replay_wave_count > 1:
            replay_desc += f" (wave {replay_wave_idx + 1}/{replay_wave_count})"
        wave_summaries = _run_eval_rollout(
            failure_args,
            env_name,
            worker_env_kwargs,
            total_steps,
            replay_desc,
            len(replay_pairs),
            policy=policy,
            recorded_agents_per_batch=recorded_agents_per_batch,
            replay_output_dir=replay_output_dir,
            capture_observations=capture_observations,
            episode_id_offset=len(summaries),
            evaluation_policy_cache=evaluation_policy_cache,
        )
        summaries.extend(wave_summaries)
    summary = drive_benchmark._write_eval_reports(summaries, failures_dir, len(pairs))
    _render_eval_replays(summaries, failures_dir)
    return {
        "episodes": summaries,
        "summary": summary,
    }
