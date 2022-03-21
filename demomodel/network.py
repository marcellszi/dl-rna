import demomodel.config as config
import demomodel.utils as utils

import torch.nn as nn
import torch
import time
import json
import tqdm
import sys


class DemonstrativeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 256, 3, padding="same")
        self.conv2 = nn.Conv1d(256, 256, 3, padding="same")

        self.fc1 = nn.Linear(config.max_length * 256, 512)
        self.fc2 = nn.Linear(512, config.max_length)

        self.flatten = nn.Flatten()

        self.spatial_dropout = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.25)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.spatial_dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.spatial_dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def format_postfix(train_loss, valid=(None, None), test=(None, None)):
    s = f"train_loss={train_loss:.3f}"

    valid_loss, valid_auc = valid
    if valid_loss is not None and valid_auc is not None:
        s += f" valid_loss={valid_loss:.3f}"
        s += f" valid_auc={valid_auc:.3f}"

    test_loss, test_auc = test
    if test_loss is not None and test_auc is not None:
        s += f" test_loss={test_loss:.3f}"
        s += f" test_auc={test_auc:.3f}"

    return s


def evaluate(model, gen, criterion):
    with torch.no_grad():
        eval_loss = 0
        eval_auc = 0

        model.eval()
        for i, (metadata, x, y) in enumerate(gen):
            x, y = x.to(config.device), y.to(config.device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            eval_loss += loss.item()

            epoch_auc = 0
            for j, l in enumerate(metadata["length"]):
                epoch_auc += utils.roc_auc(y_pred[j, :l], y[j, :l])
            eval_auc += epoch_auc / (j + 1)

    return eval_loss / (i + 1), eval_auc / (i + 1)


def fit(
    model,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    save_path=None,
    criterion=None,
    optimizer=None,
    max_epochs=1024,
    patience=10,
):
    early_stop = 0
    min_validation = float("inf")

    if criterion is None:
        criterion = nn.BCELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    training_statistics = {
        "train_loss": [],
        "train_auc": [],
        "valid_loss": [],
        "valid_auc": [],
        "test_loss": [],
        "test_auc": [],
    }

    for epoch in range(max_epochs):
        start_time = time.time()

        # train batches
        model.train()
        train_loss = 0
        with tqdm.tqdm(total=len(train_dataloader), unit="batch") as tepoch:
            for i, (_, x_train, y_train) in enumerate(train_dataloader):
                tepoch.set_description(f" {epoch:4d}")
                tepoch.update(1)
                x_train, y_train = x_train.to(config.device), y_train.to(config.device)

                optimizer.zero_grad()
                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                tepoch.set_postfix_str(format_postfix(train_loss / (i + 1)))

            train_loss, train_auc = evaluate(model, train_dataloader, criterion)
            valid_loss, valid_auc = evaluate(model, valid_dataloader, criterion)

            training_statistics["train_loss"].append(train_loss)
            training_statistics["train_auc"].append(train_auc)
            training_statistics["valid_loss"].append(valid_loss)
            training_statistics["valid_auc"].append(valid_auc)

            if test_dataloader is not None:
                test_loss, test_auc = evaluate(model, test_dataloader, criterion)
                training_statistics["test_loss"].append(test_loss)
                training_statistics["test_auc"].append(test_auc)

            else:
                test_loss, test_auc = None, None

            tepoch.set_postfix_str(
                format_postfix(
                    train_loss, (valid_loss, valid_auc), (test_loss, test_auc)
                )
            )

            # early stop
            if valid_loss > min_validation:
                early_stop += 1
                if early_stop == patience:
                    tepoch.close()
                    print("Early stopping.")
                    break
            else:
                min_validation = valid_loss
                early_stop = 0
                if save_path is not None:
                    torch.save(model, save_path / "model.pt")

        if save_path is not None:
            with open(save_path / "training_statistics.json", "w") as f:
                json.dump(training_statistics, f)
