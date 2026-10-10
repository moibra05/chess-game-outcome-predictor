class ColumnKeys:
  WHITE_ELO = "white_elo"
  BLACK_ELO = "black_elo"
  MOVE_NUMBER = "move_number"
  FEN = "fen"
  TIME_CONTROL = "time_control"
  RESULT = "result"
  SITE = "site"
  ELO_DIFF = "elo_diff"

  WHITE_PAWNS = "white_pawns"
  BLACK_PAWNS = "black_pawns"
  WHITE_KNIGHTS = "white_knights"
  BLACK_KNIGHTS = "black_knights"
  WHITE_BISHOPS = "white_bishops"
  BLACK_BISHOPS = "black_bishops"
  WHITE_ROOKS = "white_rooks"
  BLACK_ROOKS = "black_rooks"
  WHITE_QUEENS = "white_queens"
  BLACK_QUEENS = "black_queens"

  MATERIAL_DIFF = "material_diff"
  SIDE_TO_MOVE = "side_to_move"

  WHITE_CAN_CASTLE_KINGSIDE = "white_can_castle_kingside"
  WHITE_CAN_CASTLE_QUEENSIDE = "white_can_castle_queenside"
  BLACK_CAN_CASTLE_KINGSIDE = "black_can_castle_kingside"
  BLACK_CAN_CASTLE_QUEENSIDE = "black_can_castle_queenside"

  IN_CHECK = "in_check"
  LEGAL_MOVE_COUNT = "legal_move_count"

  # CNN Keys
  WHITE_PAWN_SQUARES = "white_pawn_squares"
  BLACK_PAWN_SQUARES = "black_pawn_squares"
  WHITE_KNIGHT_SQUARES = "white_knight_squares"
  BLACK_KNIGHT_SQUARES = "black_knight_squares"
  WHITE_BISHOP_SQUARES = "white_bishop_squares"
  BLACK_BISHOP_SQUARES = "black_bishop_squares"
  WHITE_ROOK_SQUARES = "white_rook_squares"
  BLACK_ROOK_SQUARES = "black_rook_squares"
  WHITE_QUEEN_SQUARES = "white_queen_squares"
  BLACK_QUEEN_SQUARES = "black_queen_squares"
  WHITE_KING_SQUARES = "white_king_squares"
  BLACK_KING_SQUARES = "black_king_squares"

BOARD_CHANNELS = [
  ColumnKeys.WHITE_PAWN_SQUARES,
  ColumnKeys.WHITE_KNIGHT_SQUARES,
  ColumnKeys.WHITE_BISHOP_SQUARES,
  ColumnKeys.WHITE_ROOK_SQUARES,
  ColumnKeys.WHITE_QUEEN_SQUARES,
  ColumnKeys.WHITE_KING_SQUARES,

  ColumnKeys.BLACK_PAWN_SQUARES,
  ColumnKeys.BLACK_KNIGHT_SQUARES,
  ColumnKeys.BLACK_BISHOP_SQUARES,
  ColumnKeys.BLACK_ROOK_SQUARES,
  ColumnKeys.BLACK_QUEEN_SQUARES,
  ColumnKeys.BLACK_KING_SQUARES,
]