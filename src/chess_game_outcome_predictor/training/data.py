import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.constants import ColumnKeys, BOARD_CHANNELS

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

METADATA_COLUMNS = [
  ColumnKeys.WHITE_ELO,
  ColumnKeys.BLACK_ELO,
  ColumnKeys.ELO_DIFF,
  ColumnKeys.MOVE_NUMBER,
  ColumnKeys.MATERIAL_DIFF,
  ColumnKeys.SIDE_TO_MOVE,
  ColumnKeys.WHITE_CAN_CASTLE_KINGSIDE,
  ColumnKeys.WHITE_CAN_CASTLE_QUEENSIDE,
  ColumnKeys.BLACK_CAN_CASTLE_KINGSIDE,
  ColumnKeys.BLACK_CAN_CASTLE_QUEENSIDE,
  ColumnKeys.IN_CHECK,
  ColumnKeys.LEGAL_MOVE_COUNT,
]

def _load_positions(path):
  df = pd.read_csv(path)

  df[ColumnKeys.RESULT] = df[ColumnKeys.RESULT].replace(
    {"1-0": "0", "0-1": "1", "1/2-1/2": "2"}
  ).astype(int)

  elo_difference = df[ColumnKeys.WHITE_ELO] - df[ColumnKeys.BLACK_ELO]
  df[ColumnKeys.ELO_DIFF] = elo_difference.astype(int)
  return df


def load_training_data(path="data/processed/game_data.csv"):
  df = _load_positions(path)
  unique_games = df[ColumnKeys.SITE].unique()

  train_games, remaining_games = train_test_split(
    unique_games, train_size=0.7, random_state=42
  )
  validation_games, test_games = train_test_split(
    remaining_games, train_size=0.5, random_state=42
  )

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


def load_cnn_training_data(path="data/processed/game_data.csv"):
  df = _load_positions(path)
  unique_games = df[ColumnKeys.SITE].unique()

  train_games, remaining_games = train_test_split(
    unique_games, train_size=0.7, random_state=42
  )
  validation_games, test_games = train_test_split(
    remaining_games, train_size=0.5, random_state=42
  )

  train_df = df[df[ColumnKeys.SITE].isin(train_games)].copy()
  validation_df = df[df[ColumnKeys.SITE].isin(validation_games)].copy()
  test_df = df[df[ColumnKeys.SITE].isin(test_games)].copy()

  test_df["raw_move_number"] = test_df[ColumnKeys.MOVE_NUMBER]

  scaler = StandardScaler()

  train_df[METADATA_COLUMNS] = scaler.fit_transform(train_df[METADATA_COLUMNS])
  validation_df[METADATA_COLUMNS] = scaler.transform(validation_df[METADATA_COLUMNS])
  test_df[METADATA_COLUMNS] = scaler.transform(test_df[METADATA_COLUMNS])

  return train_df, validation_df, test_df
