from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def plot_runs(runs, plots_dir: str | Path) -> None:
  """Plot the last saved run for each model name."""
  runs = list({run["run_name"]: run for run in runs}.values())
  labels = [run["run_name"] for run in runs]
  plots_dir = Path(plots_dir)
  plots_dir.mkdir(parents=True, exist_ok=True)

  comparisons = (
    ("accuracy.png", "Accuracy by model/run", "Accuracy", (
      ("train_accuracy", "Train"),
      ("validation_accuracy", "Validation"),
      ("test_accuracy", "Test"),
    )),
    ("log_loss.png", "Log loss by model/run", "Log loss (lower is better)", (
      ("validation_log_loss", "Validation"),
      ("test_log_loss", "Test"),
    )),
  )

  for filename, title, ylabel, series in comparisons:
    figure = Figure(figsize=(max(8, len(runs) * 1.4), 5), layout="constrained")
    FigureCanvasAgg(figure)
    try:
      ax = figure.subplots()
      width = 0.8 / len(series)

      for index, (metric, legend) in enumerate(series):
        positions = [x + (index - (len(series) - 1) / 2) * width
                     for x in range(len(runs))]
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
