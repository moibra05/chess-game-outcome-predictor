import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.build_dataset import build_dataset
from utils.constants import ColumnKeys

ELO_DIFF_CUTOFF = 100

if __name__ == "__main__":
  build_dataset(max_games=100000)

  df = pd.read_csv("data/processed/game_data.csv")

  df[ColumnKeys.RESULT] = df[ColumnKeys.RESULT].replace("1-0", "0")
  df[ColumnKeys.RESULT] = df[ColumnKeys.RESULT].replace("0-1", "1")
  df[ColumnKeys.RESULT] = df[ColumnKeys.RESULT].replace("1/2-1/2", "2")

  df[ColumnKeys.ELO_DIFF] = df[ColumnKeys.WHITE_ELO] - df[ColumnKeys.BLACK_ELO]
  df[ColumnKeys.ELO_DIFF] = df[ColumnKeys.ELO_DIFF].astype(int)

  unique_games = df[ColumnKeys.SITE].unique()

  train_games, temp = train_test_split(unique_games, train_size=0.7)
  validation_games, test_games = train_test_split(temp, train_size=0.5)

  train_df = df[df[ColumnKeys.SITE].isin(train_games)]
  validation_df = df[df[ColumnKeys.SITE].isin(validation_games)]
  test_df = df[df[ColumnKeys.SITE].isin(test_games)]

  feature_columns = [
    ColumnKeys.WHITE_ELO,
    ColumnKeys.BLACK_ELO,
    ColumnKeys.ELO_DIFF,
    ColumnKeys.MOVE_NUMBER,

    ColumnKeys.WHITE_PAWNS,
    ColumnKeys.BLACK_PAWNS,
    ColumnKeys.WHITE_KNIGHTS,
    ColumnKeys.BLACK_KNIGHTS,
    ColumnKeys.WHITE_BISHOPS,
    ColumnKeys.BLACK_BISHOPS,
    ColumnKeys.WHITE_ROOKS,
    ColumnKeys.BLACK_ROOKS,
    ColumnKeys.WHITE_QUEENS,
    ColumnKeys.BLACK_QUEENS,

    ColumnKeys.MATERIAL_DIFF,
    ColumnKeys.SIDE_TO_MOVE,

    ColumnKeys.WHITE_CAN_CASTLE_KINGSIDE,
    ColumnKeys.WHITE_CAN_CASTLE_QUEENSIDE,
    ColumnKeys.BLACK_CAN_CASTLE_KINGSIDE,
    ColumnKeys.BLACK_CAN_CASTLE_QUEENSIDE,

    ColumnKeys.IN_CHECK,
    ColumnKeys.LEGAL_MOVE_COUNT,
  ]

  train_X = train_df[feature_columns]
  train_Y = train_df[ColumnKeys.RESULT]

  validation_X = validation_df[feature_columns]
  validation_Y = validation_df[ColumnKeys.RESULT]

  test_X = test_df[feature_columns]
  test_Y = test_df[ColumnKeys.RESULT]

  model = LogisticRegression(max_iter=10000)
  model.fit(train_X, train_Y)

  train_predictions = model.predict(train_X)
  validation_predictions = model.predict(validation_X)
  test_predictions = model.predict(test_X)

  print("Train accuracy:", accuracy_score(train_Y, train_predictions))
  print("Validation accuracy:", accuracy_score(validation_Y, validation_predictions))
  print("Test accuracy:", accuracy_score(test_Y, test_predictions))
