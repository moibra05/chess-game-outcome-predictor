from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chess_game_outcome_predictor.experiment_tracking import track_run
from chess_game_outcome_predictor.training.data import load_training_data
from utils.build_dataset import build_dataset

ELO_DIFF_CUTOFF = 100

if __name__ == "__main__":
  build_dataset(max_games=100000)
  train_X, train_Y, validation_X, validation_Y, test_X, test_Y = load_training_data()

  model = Pipeline(
    [
      ("scaler", StandardScaler()),
      ("classifier", LogisticRegression(max_iter=10000)),
    ]
  )
  model.fit(train_X, train_Y)

  train_predictions = model.predict(train_X)
  validation_predictions = model.predict(validation_X)
  test_predictions = model.predict(test_X)

  train_accuracy = accuracy_score(train_Y, train_predictions)
  validation_accuracy = accuracy_score(validation_Y, validation_predictions)
  test_accuracy = accuracy_score(test_Y, test_predictions)
  validation_log_loss = log_loss(
    validation_Y,
    model.predict_proba(validation_X),
    labels=model.named_steps["classifier"].classes_,
  )
  test_log_loss = log_loss(
    test_Y, model.predict_proba(test_X), labels=model.named_steps["classifier"].classes_
  )

  print("Train accuracy:", train_accuracy)
  print("Validation accuracy:", validation_accuracy)
  print("Test accuracy:", test_accuracy)
  print("Validation log loss:", validation_log_loss)
  print("Test log loss:", test_log_loss)

  track_run(
    "Logistic regression",
    train_accuracy,
    validation_accuracy,
    test_accuracy,
    validation_log_loss,
    test_log_loss,
  )
