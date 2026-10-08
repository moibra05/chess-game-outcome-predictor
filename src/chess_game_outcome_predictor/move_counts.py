from sklearn.metrics import accuracy_score, log_loss

from utils.constants import ColumnKeys

MOVE_GROUPS = (
  ("<10", 1, 9),
  ("10–20", 10, 20),
  ("21–30", 21, 30),
  ("31–40", 31, 40),
  ("41+", 41, None),
)


def evaluate_move_counts(model, test_X, test_Y):
  move_numbers = (test_X[ColumnKeys.MOVE_NUMBER] + 1) // 2
  predictions = model.predict(test_X)
  probabilities = model.predict_proba(test_X)
  results = []

  for label, lower, upper in MOVE_GROUPS:
    mask = move_numbers >= lower

    if upper is not None:
      mask &= move_numbers <= upper

    count = int(mask.sum())
    results.append(
      {
        "move_group": label,
        "position_count": count,
        "test_accuracy": accuracy_score(test_Y[mask], predictions[mask])
        if count
        else None,
        "test_log_loss": log_loss(
          test_Y[mask],
          probabilities[mask],
          labels=model.classes_,
        )
        if count
        else None,
      }
    )

  return results
