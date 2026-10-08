from sklearn.metrics import accuracy_score, log_loss

from chess_game_outcome_predictor.experiment_tracking import track_run
from chess_game_outcome_predictor.move_counts import evaluate_move_counts


def evaluate_and_track(
  run_name, model, train_X, train_Y, validation_X, validation_Y, test_X, test_Y
):
  metrics = {
    "train_accuracy": accuracy_score(train_Y, model.predict(train_X)),
    "validation_accuracy": accuracy_score(validation_Y, model.predict(validation_X)),
    "test_accuracy": accuracy_score(test_Y, model.predict(test_X)),
    "validation_log_loss": log_loss(
      validation_Y,
      model.predict_proba(validation_X),
      labels=model.classes_,
    ),
    "test_log_loss": log_loss(
      test_Y,
      model.predict_proba(test_X),
      labels=model.classes_,
    ),
  }

  for name, value in metrics.items():
    print(f"{name}: {value:.5f}")

  move_metrics = evaluate_move_counts(model, test_X, test_Y)
  for group in move_metrics:
    label, count = group["move_group"], group["position_count"]
    if count:
      print(
        f"Moves {label}: {count} positions, "
        f"accuracy = {group['test_accuracy']:.5f}, "
        f"log loss = {group['test_log_loss']:.5f}"
      )
    else:
      print(f"Moves {label}: no test positions")

  track_run(
    run_name,
    metrics["train_accuracy"],
    metrics["validation_accuracy"],
    metrics["test_accuracy"],
    metrics["validation_log_loss"],
    metrics["test_log_loss"],
    move_metrics=move_metrics,
  )
