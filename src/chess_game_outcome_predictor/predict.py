import argparse
from pathlib import Path

import chess
import numpy as np
import pandas as pd
from joblib import load

from chess_game_outcome_predictor.training.data import FEATURE_COLUMNS
from utils.build_dataset import extra_board_features
from utils.constants import BOARD_CHANNELS, ColumnKeys

MODEL_PATHS = {
  "logistic": "logistic_model.joblib",
  "xgboost": "xgboost_model.joblib",
  "cnn": "best_cnn_model.pt",
}
RESULTS = {0: "White wins (1-0)", 1: "Black wins (0-1)", 2: "Draw (1/2-1/2)"}


def position_features(fen, white_elo, black_elo):
  board = chess.Board(fen)
  if not board.is_valid():
    raise ValueError("FEN must describe a valid chess position.")
  if white_elo < 0 or black_elo < 0:
    raise ValueError("Elo ratings must be non-negative.")
  return pd.DataFrame(
    [
      {
        ColumnKeys.WHITE_ELO: white_elo,
        ColumnKeys.BLACK_ELO: black_elo,
        ColumnKeys.ELO_DIFF: white_elo - black_elo,
        # Training counts individual half-moves, rather than full moves.
        ColumnKeys.MOVE_NUMBER: board.ply(),
        **extra_board_features(board),
      }
    ]
  )


def cnn_probabilities(features, model_path):
  import torch

  from chess_game_outcome_predictor.train_nn import METADATA_COLUMNS, ChessBoardNet
  from chess_game_outcome_predictor.training.device import DEVICE

  model = ChessBoardNet().to(DEVICE)
  model.load_state_dict(torch.load(model_path, map_location=DEVICE))
  model.eval()

  bitboards = features[BOARD_CHANNELS].to_numpy(dtype=np.uint64)
  shifts = np.arange(64, dtype=np.uint64)
  boards = ((bitboards[:, :, None] >> shifts) & 1).astype(np.float32)
  boards = torch.from_numpy(boards.reshape(1, 12, 8, 8)).to(DEVICE)

  scaler = load(Path(model_path).with_name("cnn_scaler.joblib"))
  columns = scaler.feature_names_in_
  features[columns] = scaler.transform(features[columns])

  metadata = torch.tensor(
    features[METADATA_COLUMNS].to_numpy(), dtype=torch.float32, device=DEVICE
  )

  with torch.inference_mode():
    return model(boards, metadata).softmax(dim=1)[0].cpu().numpy()


def predict(model_name, fen, white_elo, black_elo):
  features = position_features(fen, white_elo, black_elo)
  model_path = Path(MODEL_PATHS[model_name])

  if not model_path.is_file():
    raise FileNotFoundError(f"Model file not found: {model_path}. Train the model first.")

  if model_name == "cnn":
    probabilities = cnn_probabilities(features, model_path)
    classes = np.arange(3)
  else:
    model = load(model_path)
    probabilities = model.predict_proba(features[FEATURE_COLUMNS])[0]
    classes = model.classes_

  best = int(np.argmax(probabilities))
  return RESULTS[int(classes[best])], float(probabilities[best])


def main():
  parser = argparse.ArgumentParser(description="Predict a chess game's eventual result.")
  parser.add_argument("--model", required=True, choices=MODEL_PATHS)
  parser.add_argument("--fen", required=True, help="FEN position, enclosed in quotes")
  parser.add_argument("--white-elo", required=True, type=int)
  parser.add_argument("--black-elo", required=True, type=int)

  args = parser.parse_args()

  try:
    result, confidence = predict(args.model, args.fen, args.white_elo, args.black_elo)
  except (OSError, ValueError) as exc:
    parser.error(str(exc))

  print(f"Prediction: {result}")
  print(f"Confidence (model probability): {confidence:.1%}")


if __name__ == "__main__":
  main()
