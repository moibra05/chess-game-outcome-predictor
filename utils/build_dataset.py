import csv
import json
import os
import random
import tempfile
import time
from pathlib import Path
from compression import zstd
from collections import Counter

import chess.pgn
from utils.constants import ColumnKeys, BOARD_CHANNELS

fieldnames = [
  ColumnKeys.WHITE_ELO,
  ColumnKeys.BLACK_ELO,
  ColumnKeys.MOVE_NUMBER,
  ColumnKeys.FEN,
  ColumnKeys.TIME_CONTROL,
  ColumnKeys.RESULT,
  ColumnKeys.SITE,
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
  ColumnKeys.WHITE_PAWN_SQUARES,
  ColumnKeys.BLACK_PAWN_SQUARES,
  ColumnKeys.WHITE_KNIGHT_SQUARES,
  ColumnKeys.BLACK_KNIGHT_SQUARES,
  ColumnKeys.WHITE_BISHOP_SQUARES,
  ColumnKeys.BLACK_BISHOP_SQUARES,
  ColumnKeys.WHITE_ROOK_SQUARES,
  ColumnKeys.BLACK_ROOK_SQUARES,
  ColumnKeys.WHITE_QUEEN_SQUARES,
  ColumnKeys.BLACK_QUEEN_SQUARES,
  ColumnKeys.WHITE_KING_SQUARES,
  ColumnKeys.BLACK_KING_SQUARES,
]

input_path = Path("data/raw/lichess_db_standard_rated_2016-03.pgn.zst")
file_path = Path("data/processed/game_data.csv")
cache_path = file_path.with_name(f"{file_path.name}.cache.json")
CACHE_VERSION = 1


def extra_board_features(board):
    white_pawns = sum(board.pieces(chess.PAWN, chess.WHITE).tolist())
    white_bishops = sum(board.pieces(chess.BISHOP, chess.WHITE).tolist())
    white_knights = sum(board.pieces(chess.KNIGHT, chess.WHITE).tolist())
    white_rooks = sum(board.pieces(chess.ROOK, chess.WHITE).tolist())
    white_queens = sum(board.pieces(chess.QUEEN, chess.WHITE).tolist())

    black_pawns = sum(board.pieces(chess.PAWN, chess.BLACK).tolist())
    black_bishops = sum(board.pieces(chess.BISHOP, chess.BLACK).tolist())
    black_knights = sum(board.pieces(chess.KNIGHT, chess.BLACK).tolist())
    black_rooks = sum(board.pieces(chess.ROOK, chess.BLACK).tolist())
    black_queens = sum(board.pieces(chess.QUEEN, chess.BLACK).tolist())

    return {
      ColumnKeys.WHITE_PAWNS: white_pawns,
      ColumnKeys.WHITE_BISHOPS: white_bishops,
      ColumnKeys.WHITE_KNIGHTS: white_knights,
      ColumnKeys.WHITE_ROOKS: white_rooks,
      ColumnKeys.WHITE_QUEENS: white_queens,

      ColumnKeys.BLACK_PAWNS: black_pawns,
      ColumnKeys.BLACK_BISHOPS: black_bishops,
      ColumnKeys.BLACK_KNIGHTS: black_knights,
      ColumnKeys.BLACK_ROOKS: black_rooks,
      ColumnKeys.BLACK_QUEENS: black_queens,

      ColumnKeys.MATERIAL_DIFF: (
        white_pawns * 1 + white_knights * 3 + white_bishops * 3 + white_rooks * 5 + white_queens * 9
        - black_pawns * 1 - black_knights * 3 - black_bishops * 3 - black_rooks * 5 - black_queens * 9
      ),
      ColumnKeys.SIDE_TO_MOVE: int(board.turn),

      ColumnKeys.WHITE_CAN_CASTLE_KINGSIDE: int(board.has_kingside_castling_rights(chess.WHITE)),
      ColumnKeys.WHITE_CAN_CASTLE_QUEENSIDE: int(board.has_queenside_castling_rights(chess.WHITE)),
      ColumnKeys.BLACK_CAN_CASTLE_KINGSIDE: int(board.has_kingside_castling_rights(chess.BLACK)),
      ColumnKeys.BLACK_CAN_CASTLE_QUEENSIDE: int(board.has_queenside_castling_rights(chess.BLACK)),
      ColumnKeys.IN_CHECK: int(board.is_check()),
      ColumnKeys.LEGAL_MOVE_COUNT: board.legal_moves.count(),

      ColumnKeys.WHITE_PAWN_SQUARES: board.pieces(chess.PAWN, chess.WHITE).mask,
      ColumnKeys.BLACK_PAWN_SQUARES: board.pieces(chess.PAWN, chess.BLACK).mask,
      ColumnKeys.WHITE_KNIGHT_SQUARES: board.pieces(chess.KNIGHT, chess.WHITE).mask,
      ColumnKeys.BLACK_KNIGHT_SQUARES: board.pieces(chess.KNIGHT, chess.BLACK).mask,
      ColumnKeys.WHITE_BISHOP_SQUARES: board.pieces(chess.BISHOP, chess.WHITE).mask,
      ColumnKeys.BLACK_BISHOP_SQUARES: board.pieces(chess.BISHOP, chess.BLACK).mask,
      ColumnKeys.WHITE_ROOK_SQUARES: board.pieces(chess.ROOK, chess.WHITE).mask,
      ColumnKeys.BLACK_ROOK_SQUARES: board.pieces(chess.ROOK, chess.BLACK).mask,
      ColumnKeys.WHITE_QUEEN_SQUARES: board.pieces(chess.QUEEN, chess.WHITE).mask,
      ColumnKeys.BLACK_QUEEN_SQUARES: board.pieces(chess.QUEEN, chess.BLACK).mask,
      ColumnKeys.WHITE_KING_SQUARES: board.pieces(chess.KING, chess.WHITE).mask,
      ColumnKeys.BLACK_KING_SQUARES: board.pieces(chess.KING, chess.BLACK).mask,
    } 

def file_info(path):
  stat = path.stat()
  return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_dataset(max_games=10):
  cache_key = {
    "version": CACHE_VERSION,
    "max_games": max_games,
    "input": file_info(input_path),
    "fieldnames": fieldnames,
  }

  try:
    with cache_path.open(encoding="utf-8") as cache_file:
      cached = json.load(cache_file)
    if cached.get("key") == cache_key and cached.get("output") == file_info(file_path):
      print(f"Dataset already built for {max_games} games; skipping.")
      return
  except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
    pass

  file_path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(dir=file_path.parent) as temp_dir:
    temp_output = Path(temp_dir) / file_path.name
    with (
      zstd.open(input_path, "rt", encoding="utf-8") as f,
      temp_output.open("w", newline="", encoding="utf-8") as output,
    ):
      writer = csv.DictWriter(output, fieldnames=fieldnames)
      writer.writeheader()

      print("Processing games...")
      game = chess.pgn.read_game(f)
      processed_games = 0

      while game is not None and processed_games < max_games:
        moves = list(game.mainline_moves())

        if len(moves) >= 3:
          board = game.board()
          third = len(moves) // 3

          # early = random.randint(1, third)
          # middle = random.randint(third + 1, 2 * third)
          # late = random.randint(2 * third + 1, len(moves))

          # selected_board_positions = {early, middle, late}
          
          selected_board_positions = Counter(
            random.randint(2 * third + 1, len(moves)) for _ in range(3)
          )

          last_position = max(selected_board_positions)

          for move_number, move in enumerate(moves, start=1):
            board.push(move)

            if move_number in selected_board_positions:
              data = {
                ColumnKeys.SITE: game.headers["Site"],
                ColumnKeys.WHITE_ELO: game.headers["WhiteElo"],
                ColumnKeys.BLACK_ELO: game.headers["BlackElo"],
                ColumnKeys.MOVE_NUMBER: move_number,
                ColumnKeys.FEN: board.fen(),
                ColumnKeys.TIME_CONTROL: game.headers["TimeControl"],
                ColumnKeys.RESULT: game.headers["Result"],
                **extra_board_features(board),
              }

              writer.writerow(data)

            if move_number == last_position:
              break

        processed_games += 1
        game = chess.pgn.read_game(f)

    os.replace(temp_output, file_path)
    temp_cache = Path(temp_dir) / cache_path.name
    with temp_cache.open("w", encoding="utf-8") as cache_file:
      json.dump({"key": cache_key, "output": file_info(file_path)}, cache_file)
    os.replace(temp_cache, cache_path)


if __name__ == "__main__":
  start_time = time.perf_counter()

  build_dataset()

  end_time = time.perf_counter()

  elapsed_time = end_time - start_time

  print(f"Processing time: {elapsed_time:.4f} seconds.")
