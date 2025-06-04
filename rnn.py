import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from functions import apply_transforms
from functions import get_accuracy

# Hyperparameters
batch_size = 64
num_epochs = 100

# Variables
generator = torch.Generator().manual_seed(20)
train_mean_std = (torch.tensor([0.4915, 0.4823, 0.4468]), torch.tensor([0.2023, 0.1994, 0.2010]))
val_mean_std = (torch.tensor([0.4906, 0.4813, 0.4439]), torch.tensor([0.2024, 0.1995, 0.2009]))
test_mean_std = (torch.tensor([0.4942, 0.4851, 0.4504]), torch.tensor([0.2020, 0.1991, 0.2011]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the train and test data sets
trainset = torchvision.datasets.CIFAR10(root="./data",
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root="./data",
                                       train=False,
                                       download=True,
                                       transform=transforms.ToTensor())

# Divide the train set further into train set and validation set
train_size = int(0.9 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size], generator=generator)

# Apply the transformations on the data sets
trainset.dataset.transform = apply_transforms(train_mean_std, train=True)
valset.dataset.transform = apply_transforms(val_mean_std, train=False)
testset.transform = apply_transforms(test_mean_std, train=False)

# Load the data sets
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


# Creating an RNN model with optimized hyperparameters
class SimpleRNN(nn.Module):
    def __init__(self, input_size=96, hidden_size=256, num_layers=1, num_classes=10,
                 l2=64, l3=64):
        super(SimpleRNN, self).__init__()
        # RNN
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        # MLP (non-linear projection)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, l2),
            nn.BatchNorm1d(l2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(l2, l3),
            nn.BatchNorm1d(l3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(l3, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length=32, input_size=96)
        out, _ = self.rnn(x)  # out shape: (batch_size, 32, hidden_size)
        out = out[:, -1, :]  # take the output after the last run
        out = self.mlp(out)  # (batch_size, num_classes)
        return out


# Prepare the model, loss function, optimizer, and scheduler based on minimizing the validation loss
model = SimpleRNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00022, weight_decay=0.00052)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Training the model

train_losses_epoch = []
val_losses_epoch = []

train_accuracy_epoch = []
val_accuracy_epoch = []

for epoch in range(num_epochs):
    # Training for one epoch and determining the loss of training set over this epoch
    train_losses_step = []
    model.train()
    for images, labels in train_loader:  # images: (batch_size, 3, 32, 32)
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        # Reshape images to (batch_size, 32, 96) where the corresponding rows from
        # each channel are concatenated as one row
        images = images.permute(0, 2, 3, 1)  # to (batch_size, rows, columns, channels)
        images = images.reshape(images.shape[0], images.shape[1], -1)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses_step.append(loss.item())
    train_losses_epoch.append(np.array(train_losses_step).mean())

    # Determining the loss of validation set over this epoch
    model.eval()
    val_losses_step = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.permute(0, 2, 3, 1)
            images = images.reshape(images.shape[0], images.shape[1], -1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_losses_step.append(loss.item())
        val_losses_epoch.append(np.array(val_losses_step).mean())

    scheduler.step(val_losses_epoch[epoch])

    # Determining the accuracy of the train and val datasets on the model after training for this epoch
    train_accuracy_epoch.append(get_accuracy(model, train_loader))
    val_accuracy_epoch.append(get_accuracy(model, val_loader))

    # Print the statistics after one epoch
    print(
        f"After epoch {epoch + 1}: training loss = {train_losses_epoch[epoch]}, validation loss = {val_losses_epoch[epoch]}, training accuracy = {train_accuracy_epoch[epoch]}, validation accuracy = {val_accuracy_epoch[epoch]}")

print("Finished Training")

# Plot the loss and Accuracy changes over the training process
epochs = range(1, len(train_losses_epoch) + 1)

# Plot Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses_epoch, label='Train Loss')
plt.plot(epochs, val_losses_epoch, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy_epoch, label='Train Accuracy')
plt.plot(epochs, val_accuracy_epoch, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("rnn.png", dpi=300, bbox_inches="tight")

# Save the trained model
path = "./rnn.pth"
torch.save(model.state_dict(), path)
