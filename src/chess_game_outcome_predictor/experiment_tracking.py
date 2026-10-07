import csv
import math
from datetime import datetime, timezone
from pathlib import Path

from chess_game_outcome_predictor.plotting import plot_runs

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


def track_run(
  run_name,
  train_accuracy,
  validation_accuracy,
  test_accuracy,
  validation_log_loss,
  test_log_loss,
  *,
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

  plot_runs([*runs, run], plots_dir)