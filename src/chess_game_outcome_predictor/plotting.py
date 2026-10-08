from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from chess_game_outcome_predictor.move_counts import MOVE_GROUPS


def plot_runs(runs, plots_dir: str | Path) -> None:
  """Plot the last saved run for each model name."""
  runs = list({run["run_name"]: run for run in runs}.values())
  labels = [run["run_name"] for run in runs]
  plots_dir = Path(plots_dir)
  plots_dir.mkdir(parents=True, exist_ok=True)

  comparisons = (
    (
      "accuracy.png",
      "Accuracy by model/run",
      "Accuracy",
      (
        ("train_accuracy", "Train"),
        ("validation_accuracy", "Validation"),
        ("test_accuracy", "Test"),
      ),
    ),
    (
      "log_loss.png",
      "Log loss by model/run",
      "Log loss (lower is better)",
      (
        ("validation_log_loss", "Validation"),
        ("test_log_loss", "Test"),
      ),
    ),
  )

  for filename, title, ylabel, series in comparisons:
    figure = Figure(figsize=(max(8, len(runs) * 1.4), 5), layout="constrained")
    FigureCanvasAgg(figure)
    try:
      ax = figure.subplots()
      width = 0.8 / len(series)

      for index, (metric, legend) in enumerate(series):
        positions = [
          x + (index - (len(series) - 1) / 2) * width for x in range(len(runs))
        ]
        ax.bar(positions, [float(run[metric]) for run in runs], width, label=legend)

      ax.set_xticks(range(len(runs)), labels, rotation=30, ha="right")
      ax.set(title=title, xlabel="Model / run", ylabel=ylabel)
      ax.set_axisbelow(True)
      ax.grid(axis="y", alpha=0.25)

      if filename == "accuracy.png":
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
      else:
        ax.set_ylim(bottom=0)
      ax.legend()
      figure.savefig(plots_dir / filename, dpi=160)

    finally:
      figure.clear()


def plot_move_counts(runs, plots_dir):
  plots_dir = Path(plots_dir)
  plots_dir.mkdir(parents=True, exist_ok=True)

  figure = Figure(figsize=(15, 5), layout="constrained")
  FigureCanvasAgg(figure)

  axes = figure.subplots(1, 3)
  labels = [label for label, _, _ in MOVE_GROUPS]
  model_names = list(dict.fromkeys(run["run_name"] for run in runs))

  comparisons = (
    ("test_accuracy", "Test accuracy (higher is better)", "Accuracy"),
    ("test_log_loss", "Test log loss (lower is better)", "Log loss"),
    ("position_count", "Test positions per group", "Positions"),
  )

  for ax, (metric, title, ylabel) in zip(axes, comparisons):
    for name in model_names:
      groups = {run["move_group"]: run for run in runs if run["run_name"] == name}
      values = []

      for label in labels:
        value = groups.get(label, {}).get(metric)
        values.append(float(value) if value not in (None, "") else float("nan"))

      ax.plot(range(len(labels)), values, marker="o", label=name)

    ax.set_xticks(range(len(labels)), labels)
    ax.set(title=title, xlabel="Chess move number", ylabel=ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)

    if model_names:
      ax.legend()
    else:
      ax.text(
        0.5,
        0.5,
        "No stage evaluations for the latest runs",
        ha="center",
        transform=ax.transAxes,
      )

  axes[0].set_ylim(0, 1)
  axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
  figure.savefig(plots_dir / "performance_by_move_count.png")
  figure.clear()
