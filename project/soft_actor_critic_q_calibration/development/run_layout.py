"""Paths and entry points shared by collection and analysis."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def model_directory(path):
    model_dir = Path(path).expanduser().resolve()
    if model_dir.name == "records":
        model_dir = model_dir.parent
    relative = model_dir.relative_to(PROJECT_DIR / "runs")
    if len(relative.parts) != 2:
        raise ValueError("Expected runs/<run>/<model> directory")
    return model_dir


def records_directory(path):
    return model_directory(path) / "records"


def initialize_model_directory(path):
    model_dir = model_directory(path)
    descriptions = {
        "scripts": "Run-specific analysis entry point. Run `python analyze.py --help` here after activating the repository virtual environment. Shared implementation lives in development/.\n",
        "records": "Collected probes and reference trajectories, manifests, progress, logs, and numerical summaries. Derived plot tables/counts/notes live in analysis/<plot-family>/. Historical commands and manifest paths describe the original execution; they are preserved as provenance.\n",
        "renders": "Interactive full-scene HTML rollouts. Open a replay and select a first action for car 0.\n",
        "plots": "Images grouped by analysis family: target_error_plots/, raw_q_plots/, and driving_quality_plots/. Supporting CSV/JSON files are in ../records/analysis/. Regenerate figures using ../scripts/analyze.py. Only images and one README per directory belong here.\n",
    }
    for name, description in descriptions.items():
        directory = model_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        readme = directory / "README.md"
        if not readme.exists():
            readme.write_text(f"# {name.capitalize()}\n\n{description}")
    readme = model_dir / "README.md"
    if not readme.exists():
        readme.write_text(f"# {model_dir.parent.name} / {model_dir.name}\n\n"
                          "- [Plots](plots/README.md): visual analysis.\n"
                          "- [Renders](renders/README.md): interactive rollouts.\n"
                          "- [Records](records/README.md): collected data and derived tables.\n"
                          "- [Scripts](scripts/README.md): commands for this model.\n")
    script = model_dir / "scripts" / "analyze.py"
    if not script.exists():
        script.write_text('"""Regenerate analysis for this model from saved records."""\n'
                          'from pathlib import Path\nimport sys\n\n'
                          'MODEL_DIR = Path(__file__).resolve().parents[1]\n'
                          'sys.path.insert(0, str(MODEL_DIR.parents[4]))\n'
                          'from project.soft_actor_critic_q_calibration.development.analyze_run import main\n\n'
                          'if __name__ == "__main__":\n'
                          '    main(sys.argv[1:], model_dir=MODEL_DIR)\n')
    return model_dir
