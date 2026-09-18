import csv
import os
import random
import time
from collections import Counter
from compression import zstd

import chess.pgn
from constants import BLACK_ELO, FEN, MOVE_NUMBER, RESULT, SITE, TIME_CONTROL, WHITE_ELO

fieldnames = [WHITE_ELO, BLACK_ELO, MOVE_NUMBER, FEN, TIME_CONTROL, RESULT, SITE]

file_path = "data/processed/game_data.csv"


def build_dataset(max_games=100000):
  write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0

  with (
    zstd.open(
      "data/raw/lichess_db_standard_rated_2016-03.pgn.zst", "rt", encoding="utf-8"
    ) as f,
    open(file_path, "a", newline="", encoding="utf-8") as output,
  ):
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    if write_header:
      writer.writeheader()

    print("Processing games...")
    game = chess.pgn.read_game(f)
    processed_games = 0

    while game is not None and processed_games <= max_games:
      moves = list(game.mainline_moves())

      if moves:
        board = game.board()
        selected_board_positions = Counter(
          random.randint(1, len(moves)) for _ in range(3)
        )
        last_position = max(selected_board_positions)

        for move_number, move in enumerate(moves, start=1):
          board.push(move)
          row_count = selected_board_positions.get(move_number, 0)
          if row_count:
            data = {
              SITE: game.headers["Site"],
              WHITE_ELO: game.headers["WhiteElo"],
              BLACK_ELO: game.headers["BlackElo"],
              MOVE_NUMBER: move_number,
              FEN: board.fen(),
              TIME_CONTROL: game.headers["TimeControl"],
              RESULT: game.headers["Result"],
            }
            for _ in range(row_count):
              writer.writerow(data)

          if move_number == last_position:
            break

      game = chess.pgn.read_game(f)
      processed_games += 1


if __name__ == "__main__":
  start_time = time.perf_counter()

  build_dataset()

  end_time = time.perf_counter()

  elapsed_time = end_time - start_time

  print(f"Processing time: {elapsed_time:.4f} seconds.")
