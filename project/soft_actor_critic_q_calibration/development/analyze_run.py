"""Regenerate selected figures from saved probe records; never collect rollouts."""
import argparse
from pathlib import Path


def main(argv=None, *, model_dir=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if model_dir is None:
        parser.add_argument("model_dir", type=Path)
    else:
        parser.set_defaults(model_dir=model_dir)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-probe", metavar="PROBE_ID",
                      help="Replay all first actions for one saved probe and write an HTML viewer")
    parser.add_argument("--repeat", type=int, default=0, help="Continuation repeat index for HTML replay")
    parser.add_argument("--device", default=None, help="Replay inference device (default: recorded device)")
    mode.add_argument("--figure", default="all", choices=(
        "all", "target-errors", "rank-errors", "raw-q", "rank-raw", "3d",
        "driving", "centered", "centered-zoom", "summaries"))
    args = parser.parse_args(argv)
    from .run_layout import model_directory, records_directory, initialize_model_directory
    model_dir = model_directory(args.model_dir)
    if not any((records_directory(model_dir) / "probes").glob("*.json")):
        parser.error(f"No probe records in {records_directory(model_dir) / 'probes'}")
    initialize_model_directory(model_dir)
    if args.render_probe is not None:
        import json
        from .render_capture import capture
        from .rollout_viewer import write_viewer

        probe_id = args.render_probe
        if not probe_id or Path(probe_id).name != probe_id or args.repeat < 0:
            parser.error("Expected a probe filename stem and nonnegative repeat index")
        records = records_directory(model_dir)
        if not (records / "probes" / f"{probe_id}.json").is_file():
            parser.error(f"Unknown probe {probe_id!r}; see {records / 'probes'}")
        manifest = json.loads((records / "manifest.json").read_text())
        device = args.device or manifest["checkpoint"]["audit"]["device"]
        payload = capture(model_dir, probe_id, args.repeat, device)
        directory = model_dir / "renders"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{probe_id}__repeat_{args.repeat:03d}.html"
        # Publish only after the full multi-action page has been written.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=directory) as staging:
            page = Path(staging) / destination.name
            write_viewer(payload, page)
            page.replace(destination)
        print(destination)
        return
    if args.figure == "summaries":
        from .summarize_probes import summarize
        summarize(model_dir)
        return
    from .plot_target_errors import plot_target_errors
    from .plot_grouped_target_errors import plot_grouped_target_errors
    from .plot_target_errors_3d import plot_target_errors_3d
    from .plot_driving_quality import plot_driving_quality
    actions = {
        "target-errors": lambda: plot_target_errors(model_dir),
        "rank-errors": lambda: plot_grouped_target_errors(model_dir),
        "raw-q": lambda: plot_target_errors(model_dir, raw_q=True),
        "rank-raw": lambda: plot_grouped_target_errors(model_dir, raw_q=True),
        "3d": lambda: plot_target_errors_3d(model_dir),
        "driving": lambda: plot_driving_quality(model_dir),
        "centered": lambda: plot_driving_quality(model_dir, center_target=True),
        "centered-zoom": lambda: plot_driving_quality(model_dir, center_target=True, x_limits=(-0.4, 0.2)),
    }
    for name, action in actions.items():
        if args.figure in ("all", name):
            action()


if __name__ == "__main__":
    main()
