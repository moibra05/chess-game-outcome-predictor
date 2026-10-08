from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chess_game_outcome_predictor.evaluation import evaluate_and_track
from chess_game_outcome_predictor.training.data import load_training_data
from utils.build_dataset import build_dataset

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

  evaluate_and_track(
    "Logistic regression",
    model,
    train_X,
    train_Y,
    validation_X,
    validation_Y,
    test_X,
    test_Y,
  )
