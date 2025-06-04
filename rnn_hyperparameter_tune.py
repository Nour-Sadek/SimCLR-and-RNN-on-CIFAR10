import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn

import optuna
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from functions import apply_transforms

# Variables
generator = torch.Generator().manual_seed(20)
train_mean_std = (torch.tensor([0.4915, 0.4823, 0.4468]), torch.tensor([0.2023, 0.1994, 0.2010]))
val_mean_std = (torch.tensor([0.4906, 0.4813, 0.4439]), torch.tensor([0.2024, 0.1995, 0.2009]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Get the CIFAR10 train set
trainset = torchvision.datasets.CIFAR10(root="./data",
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())

# Divide the train set further into train set and validation set
train_size = int(0.9 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size], generator=generator)

# Apply the transformations on the data sets
trainset.dataset.transform = apply_transforms(train_mean_std, train=True)
valset.dataset.transform = apply_transforms(val_mean_std, train=False)


class LitSimpleRNN(pl.LightningModule):
    def __init__(self, input_size=96, hidden_size=128, num_layers=3, num_classes=10,
                 l2=64, l3=32, learning_rate=1e-4, weight_decay=5e-4):
        super().__init__()
        self.save_hyperparameters()

        # RNN
        self.rnn = nn.RNN(input_size=input_size, hidden_size=self.hparams.hidden_size,
                          num_layers=self.hparams.num_layers, batch_first=True)
        # MLP (non-linear projection)
        self.mlp = nn.Sequential(
            nn.Linear(self.hparams.hidden_size, self.hparams.l2),
            nn.BatchNorm1d(self.hparams.l2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hparams.l2, self.hparams.l3),
            nn.BatchNorm1d(self.hparams.l3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hparams.l3, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.mlp(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )


def objective(trial):
    # Hyperparameter search space
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    num_layers = trial.suggest_categorical("num_layers", [1, 3, 5])
    l2 = trial.suggest_categorical("l2", [2 ** i for i in range(4, 7)])
    l3 = trial.suggest_categorical("l3", [2 ** i for i in range(4, 7)])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 5e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2 ** i for i in range(5, 9)])

    # Data
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = LitSimpleRNN(l2=l2, l3=l3, learning_rate=learning_rate, weight_decay=weight_decay)

    # Trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        logger=TensorBoardLogger("optuna_logs", name="cifar10"),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_loss"].item()


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Best hyperparameters:", study.best_trial.params)

# Best hyperparameters: {'hidden_size': 256, 'num_layers': 1, 'l2': 64, 'l3': 64,
# 'learning_rate': 0.00021990370923287235, 'weight_decay': 0.0005182777790955891, 'batch_size': 64}
# validation loss of trial was: 1.4926868677139282
