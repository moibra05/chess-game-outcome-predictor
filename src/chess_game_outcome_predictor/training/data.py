"""Shared tabular features and game-level 70/15/15 data splitting."""

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.constants import ColumnKeys

FEATURE_COLUMNS = [
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


def load_training_data(path="data/processed/game_data.csv"):
  """Return train_X, train_Y, validation_X, validation_Y, test_X, test_Y.

  Split whole games to keep positions from the same game in one partition.
  Splits remain random on every call, matching the original training script.
  """
  df = pd.read_csv(path)

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

  train_X = train_df[FEATURE_COLUMNS]
  train_Y = train_df[ColumnKeys.RESULT]

  validation_X = validation_df[FEATURE_COLUMNS]
  validation_Y = validation_df[ColumnKeys.RESULT]

  test_X = test_df[FEATURE_COLUMNS]
  test_Y = test_df[ColumnKeys.RESULT]

  return train_X, train_Y, validation_X, validation_Y, test_X, test_Y
