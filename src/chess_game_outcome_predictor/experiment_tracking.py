import csv
import math
from datetime import datetime, timezone
from pathlib import Path

from chess_game_outcome_predictor.plotting import plot_move_counts, plot_runs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "experiments.csv"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "plots"
METRIC_NAMES = (
  "train_accuracy",
  "validation_accuracy",
  "test_accuracy",
  "validation_log_loss",
  "test_log_loss",
)
FIELDNAMES = ("run_name", "timestamp", *METRIC_NAMES)
MOVE_FIELDNAMES = (
  "run_name",
  "timestamp",
  "move_group",
  "position_count",
  "test_accuracy",
  "test_log_loss",
)


def track_run(
  run_name,
  train_accuracy,
  validation_accuracy,
  test_accuracy,
  validation_log_loss,
  test_log_loss,
  move_metrics=None,
  results_path=DEFAULT_RESULTS_PATH,
  plots_dir=DEFAULT_PLOTS_DIR,
):

  if not isinstance(run_name, str) or not run_name.strip():
    raise ValueError("run_name must be a nonempty string")

  values = (
    train_accuracy,
    validation_accuracy,
    test_accuracy,
    validation_log_loss,
    test_log_loss,
  )
  metrics = dict(zip(METRIC_NAMES, map(float, values)))

  for name, value in metrics.items():
    if not math.isfinite(value) or value < 0:
      raise ValueError(f"{name} must be finite and nonnegative")
    if name.endswith("accuracy") and value > 1:
      raise ValueError(f"{name} must be a fraction between 0 and 1")

  results_path = Path(results_path)
  results_path.parent.mkdir(parents=True, exist_ok=True)

  # logs run in results/experiment.csv
  with results_path.open("a+", newline="", encoding="utf-8") as handle:
    handle.seek(0)
    reader = csv.DictReader(handle)

    if reader.fieldnames is not None and reader.fieldnames != list(FIELDNAMES):
      raise ValueError(f"Unexpected CSV columns in {results_path}")

    runs = list(reader)
    run = {
      "run_name": run_name,
      "timestamp": datetime.now(timezone.utc).isoformat(),
      **metrics,
    }
    handle.seek(0, 2)
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)

    if handle.tell() == 0:
      writer.writeheader()
    writer.writerow(run)

  runs.append(run)
  plot_runs(runs, plots_dir)

  move_results_path = results_path.with_name(f"{results_path.stem}_by_move_count.csv")

  if move_metrics is not None:
    with move_results_path.open("a+", newline="", encoding="utf-8") as handle:
      handle.seek(0, 2)
      writer = csv.DictWriter(handle, fieldnames=MOVE_FIELDNAMES)

      if handle.tell() == 0:
        writer.writeheader()

      writer.writerows(
        {
          "run_name": run_name,
          "timestamp": run["timestamp"],
          **group,
        }
        for group in move_metrics
      )

  if move_results_path.exists():
    latest_timestamps = {saved["run_name"]: saved["timestamp"] for saved in runs}
    with move_results_path.open(newline="", encoding="utf-8") as handle:
      move_runs = [
        row
        for row in csv.DictReader(handle)
        if row["timestamp"] == latest_timestamps.get(row["run_name"])
      ]

    plot_move_counts(move_runs, plots_dir)
