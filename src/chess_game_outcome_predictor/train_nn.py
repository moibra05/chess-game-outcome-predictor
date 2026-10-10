import numpy as np
import torch
import torch.nn.functional as F
from joblib import dump
from sklearn.metrics import accuracy_score, log_loss
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from chess_game_outcome_predictor.experiment_tracking import track_run
from chess_game_outcome_predictor.move_counts import MOVE_GROUPS
from chess_game_outcome_predictor.training.data import load_cnn_training_data
from chess_game_outcome_predictor.training.device import DEVICE
from utils.build_dataset import build_dataset
from utils.constants import BOARD_CHANNELS, ColumnKeys

METADATA_COLUMNS = [
  ColumnKeys.WHITE_ELO,
  ColumnKeys.BLACK_ELO,
  ColumnKeys.ELO_DIFF,
  ColumnKeys.MOVE_NUMBER,
  ColumnKeys.SIDE_TO_MOVE,
]


class ChessDataset(Dataset):
  def __init__(self, df):
    bitboards = df[BOARD_CHANNELS].to_numpy(dtype=np.uint64)
    shifts = np.arange(64, dtype=np.uint64)

    # unpack each bitboard in square order from a1 through h8.
    boards = ((bitboards[:, :, None] >> shifts) & 1).astype(np.uint8)
    boards = boards.reshape(-1, 12, 8, 8)

    metadata = df[METADATA_COLUMNS].to_numpy(dtype=np.float32)

    labels = df[ColumnKeys.RESULT].to_numpy(dtype=np.int64, copy=True)

    self.boards = torch.from_numpy(boards)
    self.metadata = torch.from_numpy(metadata)
    self.labels = torch.from_numpy(labels)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, i):
    return self.boards[i], self.metadata[i], self.labels[i]


class ChessBoardNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(64 * 8 * 8 + len(METADATA_COLUMNS), 128)
    self.fc2 = nn.Linear(128, 3)
    self.dropout = nn.Dropout(0.3)

  def forward(self, board, metadata):
    x = F.relu(self.conv1(board))
    x = F.relu(self.conv2(x))

    x = torch.flatten(x, start_dim=1)
    x = torch.cat((x, metadata), dim=1)

    x = F.relu(self.fc1(x))
    x = self.dropout(x)

    return self.fc2(x)


def _batch_to_device(boards, metadata, labels):
  boards = boards.to(DEVICE, dtype=torch.float32, non_blocking=True)
  metadata = metadata.to(DEVICE, non_blocking=True)
  labels = labels.to(DEVICE, non_blocking=True)
  return boards, metadata, labels


def train_epoch(model, loader, criterion, optimizer):
  model.train()
  running_loss = 0.0

  for boards, metadata, labels in loader:
    optimizer.zero_grad()
    boards, metadata, labels = _batch_to_device(boards, metadata, labels)
    logits = model(boards, metadata)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  return running_loss / len(loader)


def evaluate_model(model, loader, criterion):
  with torch.no_grad():
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for boards, metadata, labels in loader:
      boards, metadata, labels = _batch_to_device(boards, metadata, labels)
      logits = model(boards, metadata)
      loss = criterion(logits, labels)
      total_loss += loss.item()

      predicted_classes = logits.argmax(dim=1)
      correct += (predicted_classes == labels).sum().item()
      total += labels.size(0)

  return total_loss / len(loader), correct / total


def predict_probabilities(model, loader):
  model.eval()
  probabilities = []
  with torch.no_grad():
    for boards, metadata, labels in loader:
      boards, metadata, _ = _batch_to_device(boards, metadata, labels)
      logits = model(boards, metadata)
      probabilities.append(logits.softmax(dim=1).cpu().numpy())
  return np.concatenate(probabilities)


def evaluate_and_track_nn(model, train_dataset, validation_loader, test_loader, test_df):
  train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=4)
  train_probabilities = predict_probabilities(model, train_loader)

  validation_probabilities = predict_probabilities(model, validation_loader)
  test_probabilities = predict_probabilities(model, test_loader)

  train_labels = train_dataset.labels.numpy()
  validation_labels = validation_loader.dataset.labels.numpy()
  test_labels = test_loader.dataset.labels.numpy()

  classes = np.arange(3)

  metrics = {
    "train_accuracy": accuracy_score(train_labels, train_probabilities.argmax(axis=1)),
    "validation_accuracy": accuracy_score(
      validation_labels, validation_probabilities.argmax(axis=1)
    ),
    "test_accuracy": accuracy_score(test_labels, test_probabilities.argmax(axis=1)),
    "validation_log_loss": log_loss(
      validation_labels, validation_probabilities, labels=classes
    ),
    "test_log_loss": log_loss(test_labels, test_probabilities, labels=classes),
  }
  for name, value in metrics.items():
    print(f"{name}: {value:.5f}")

  test_move_metrics = evaluate_nn_move_counts(test_df, test_labels, test_probabilities)
  track_run(
    "Neural network",
    **metrics,
    move_metrics=test_move_metrics,
  )


def evaluate_nn_move_counts(df, labels, probabilities):
  move_numbers = (df["raw_move_number"].to_numpy() + 1) // 2
  move_metrics = []

  for label, lower, upper in MOVE_GROUPS:
    mask = move_numbers >= lower

    if upper is not None:
      mask &= move_numbers <= upper

    count = int(mask.sum())

    group = {
      "move_group": label,
      "position_count": count,
      "test_accuracy": accuracy_score(labels[mask], probabilities[mask].argmax(axis=1))
      if count
      else None,
      "test_log_loss": log_loss(labels[mask], probabilities[mask], labels=np.arange(3))
      if count
      else None,
    }

    move_metrics.append(group)

    if count:
      print(
        f"Test moves {label}: {count} positions, "
        f"accuracy = {group['test_accuracy']:.5f}, "
        f"log loss = {group['test_log_loss']:.5f}"
      )
    else:
      print(f"Test moves {label}: no positions")

  return move_metrics


def main():
  build_dataset(max_games=100000)
  train_df, validation_df, test_df, scaler = load_cnn_training_data()

  train_dataset = ChessDataset(train_df)
  validation_dataset = ChessDataset(validation_df)
  test_dataset = ChessDataset(test_df)

  train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=4,
  )
  validation_loader = DataLoader(
    validation_dataset,
    batch_size=1024,
    num_workers=4,
  )
  test_loader = DataLoader(
    test_dataset,
    batch_size=1024,
    num_workers=4,
  )

  model = ChessBoardNet().to(DEVICE)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
  )
  epochs = 15
  best_validation_loss = float("inf")

  for epoch in range(epochs):
    training_loss = train_epoch(model, train_loader, criterion, optimizer)
    validation_loss, validation_accuracy = evaluate_model(
      model, validation_loader, criterion
    )

    if validation_loss < best_validation_loss:
      best_validation_loss = validation_loss
      torch.save(model.state_dict(), "best_cnn_model.pt")
      dump(scaler, "cnn_scaler.joblib")

    print(
      f"Epoch {epoch + 1}: "
      f"train loss={training_loss:.4f}, "
      f"val loss={validation_loss:.4f}, "
      f"val acc={validation_accuracy:.4f}"
    )

  model.load_state_dict(torch.load("best_cnn_model.pt", map_location=DEVICE))
  test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)

  print("Test loss:", test_loss)
  print("Test accuracy:", test_accuracy)
  evaluate_and_track_nn(model, train_dataset, validation_loader, test_loader, test_df)


if __name__ == "__main__":
  main()
