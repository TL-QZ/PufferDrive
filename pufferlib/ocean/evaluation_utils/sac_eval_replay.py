import numbers
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

import pufferlib
import pufferlib.sac_viz
import pufferlib.viz
from pufferlib.ocean.evaluation_utils.eval_replay import EvalReplayCapture


class SACReplayCapture(EvalReplayCapture):
    """Capture SAC policy history and write value-free interactive replays."""

    def capture_frame(self, obs, policy_obs_tensor, raw_action, action, logits, value, logprob, entropy):
        if self.policy_history_frame_count == self.capture_batch_steps:
            self.policy_history = {}
            self.policy_history_frame_count = 0
        replay_frame = {
            "raw_action": np.asarray(raw_action, dtype=np.float32),
            "clipped_action": np.asarray(action, dtype=np.float32),
            "entropy": entropy[: self.agents_per_batch].detach().reshape(-1).float().cpu().numpy(),
        }
        if self.capture_observations:
            replay_frame["obs"] = np.asarray(obs, dtype=np.float16)
        if isinstance(logits, torch.distributions.Normal):
            replay_frame["policy_mean"] = logits.loc[: self.agents_per_batch].detach().float().cpu().numpy()
            replay_frame["policy_std"] = logits.scale[: self.agents_per_batch].detach().float().cpu().numpy()
            replay_frame["policy_log_prob"] = (
                logprob[: self.agents_per_batch].detach().reshape(-1).float().cpu().numpy()
            )
        else:
            discrete_logits = logits if isinstance(logits, torch.Tensor) else logits[0]
            replay_frame["policy_probs"] = (
                torch.softmax(discrete_logits[: self.agents_per_batch], dim=-1).detach().float().cpu().numpy()
            )
        if self.pool_slot_counts is not None:
            for pool_name, pool_values in self.pool_slot_counts(policy_obs_tensor).items():
                replay_frame[pool_name] = (
                    pool_values[: self.agents_per_batch].detach().cpu().numpy().astype(np.int16, copy=False)
                )

        if not self.policy_history:
            self.policy_history = {
                replay_key: np.empty(
                    (self.capture_batch_steps, *frame_values.shape),
                    dtype=frame_values.dtype,
                )
                for replay_key, frame_values in replay_frame.items()
            }
        for replay_key, frame_values in replay_frame.items():
            self.policy_history[replay_key][self.policy_history_frame_count] = frame_values
        self.policy_history_frame_count += 1

    def write_pending(self):
        scenarios, replays, replay_paths = zip(*self.pending_replays)
        writer_count = min(self.replay_writer_count, len(self.pending_replays))
        with ThreadPoolExecutor(max_workers=writer_count) as replay_writer:
            for _ in replay_writer.map(
                pufferlib.sac_viz.save_interactive_replay_zlib,
                scenarios,
                replays,
                replay_paths,
            ):
                pass
        self.pending_replays = []


def _render_eval_replays(episode_summaries, out_dir):
    """Render captured eval replays as navigable HTML pages plus an index."""
    render_dir = os.path.join(out_dir, "rendered_replays")
    os.makedirs(render_dir, exist_ok=True)

    file_metrics = {}
    replay_paths = []
    output_paths = []
    for episode_id, summary in enumerate(episode_summaries):
        replay_path = summary.get("replay_path")
        if not replay_path or not os.path.isfile(replay_path):
            raise RuntimeError(f"Cannot render episode {episode_id}: replay file is missing: {replay_path}")
        replay_filename = os.path.basename(replay_path)
        replay_suffix = ".replay.zlib"
        if not replay_filename.endswith(replay_suffix):
            raise RuntimeError(f"Cannot render episode {episode_id}: unexpected replay filename: {replay_filename}")
        html_filename = f"{replay_filename[: -len(replay_suffix)]}.html"
        output_path = os.path.join(render_dir, html_filename)
        replay_paths.append(replay_path)
        output_paths.append(output_path)
        file_metrics[html_filename] = {
            key: value for key, value in summary.items() if isinstance(value, numbers.Real) and np.isfinite(value)
        }

    if replay_paths:
        html_renderer_count = min(os.cpu_count() or 1, len(replay_paths))
        with ThreadPoolExecutor(max_workers=html_renderer_count) as html_renderer:
            rendered_replays = html_renderer.map(
                pufferlib.sac_viz.render_interactive_replay_zlib,
                replay_paths,
                output_paths,
            )
            for _ in tqdm(rendered_replays, total=len(replay_paths), desc="Rendering replay HTML"):
                pass

    pufferlib.viz.build_gallery_index(render_dir, file_metrics=file_metrics)
    print(f"Rendered {len(episode_summaries)} replay pages into {render_dir}")
    print(f"Wrote replay index to {os.path.join(render_dir, 'index.html')}")
    return render_dir
