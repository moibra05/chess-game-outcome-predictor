# Chess Game Outcome Predictor

Predict white wins, black wins, and draws from board features and player ratings
in Lichess games. Compare logistic regression and XGBoost, including whether
predictions improve as a game progresses.

## Run

Use Python 3.14+ and `uv`. Place the March 2016 rated standard games archive at
`data/raw/lichess_db_standard_rated_2016-03.pgn.zst`, then run from the project root:

```sh
uv sync
uv run python -m chess_game_outcome_predictor.train_logistic
uv run python -m chess_game_outcome_predictor.train_xgboost
```

Both models use the same game level training, validation, and test split.
XGBoost selects hyperparameters using validation log loss.

## Plots

Charts show the latest run of each model. Accuracy is better when higher;
log loss is better when lower.

![Accuracy](plots/accuracy.png)

![Log loss](plots/log_loss.png)

The stage comparison evaluates test positions at moves `<10`, `10–20`, `21–30`,
`31–40`, and `41+`. The dataset counts half moves; these
groups use chess move numbers, with two half moves per move. Counts show the
number of sampled test positions, rather than distinct games.

![Performance by move count](plots/performance_by_move_count.png)

Run history is saved in `results/experiments.csv` and
`results/experiments_by_move_count.csv`.
