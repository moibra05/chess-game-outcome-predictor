from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

from chess_game_outcome_predictor.evaluation import evaluate_and_track
from chess_game_outcome_predictor.training.data import load_training_data
from utils.build_dataset import build_dataset

PARAM_GRID = {
  "max_depth": [3, 5, 7],
  "learning_rate": [0.03, 0.1, 0.2],
}


def tune_model(train_X, train_Y, validation_X, validation_Y):
  best_model = None
  best_params = None
  best_loss = float("inf")

  for params in ParameterGrid(PARAM_GRID):
    candidate = XGBClassifier(
      objective="multi:softprob",
      num_class=3,
      n_estimators=500,
      early_stopping_rounds=30,
      random_state=42,
      **params,
    )
    candidate.fit(
      train_X,
      train_Y,
      eval_set=[(validation_X, validation_Y)],
      verbose=False,
    )
    validation_loss = log_loss(
      validation_Y,
      candidate.predict_proba(validation_X),
      labels=candidate.classes_,
    )
    print(f"{params}: validation log loss = {validation_loss:.5f}")

    if validation_loss < best_loss:
      best_model = candidate
      best_params = params
      best_loss = validation_loss

  print("Best hyperparameters:", best_params)
  print("Best boosting rounds:", best_model.best_iteration + 1)
  return best_model


if __name__ == "__main__":
  build_dataset(max_games=100000)
  train_X, train_Y, validation_X, validation_Y, test_X, test_Y = load_training_data()

  model = tune_model(train_X, train_Y, validation_X, validation_Y)

  evaluate_and_track(
    "XGBoost",
    model,
    train_X,
    train_Y,
    validation_X,
    validation_Y,
    test_X,
    test_Y,
  )
