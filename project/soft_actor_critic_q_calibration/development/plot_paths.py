"""Keep figures and their supporting records in the same model run."""
if __package__:
    from .run_layout import model_directory, records_directory
else:
    from run_layout import model_directory, records_directory


def plot_output_dir(run_dir, family, *, artifact_type="plots"):
    if family not in ("target_error_plots", "raw_q_plots", "driving_quality_plots", "geometry_plots"):
        raise ValueError(f"Unknown plot family: {family}")
    if artifact_type not in ("plots", "analysis"):
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    model_dir = model_directory(run_dir)
    base = model_dir / "plots" if artifact_type == "plots" else model_dir / "records" / "analysis"
    output = base / family
    output.mkdir(parents=True, exist_ok=True)
    readme = output / "README.md"
    if not readme.exists():
        description = (f"Images for {family}. Supporting CSV/JSON and notes are in ../../records/analysis/{family}/."
                       if artifact_type == "plots" else
                       f"Supporting tables, counts, and notes for the figures in ../../../plots/{family}/.")
        readme.write_text(f"# {family}\n\n{description}\n")
    return output
