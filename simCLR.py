import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from functions import nt_xent_loss

# Hyperparameters
batch_size = 1024
num_epochs = 500

# Variables
train_mean_std = (torch.tensor([0.4913, 0.4820, 0.4461]), torch.tensor([0.2024, 0.1994, 0.2008]))
test_mean_std = (torch.tensor([0.4942, 0.4851, 0.4504]), torch.tensor([0.2020, 0.1991, 0.2011]))

generator = torch.Generator().manual_seed(31)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Determine the accuracy of a model's predictions on a specific data set
def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    accuracy = round((correct / total) * 100, 2)
    return accuracy


# When the dataset is loaded, instead of returning inputs and labels, I want to views of the inputs to be returned
class simCLRInput(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return self.transform(x), self.transform(x)


class simCLRModel(nn.Module):
    def __init__(self, encoder, encoder_out_features_num, l2, l3):
        super().__init__()
        # Define the encoder
        self.encoder = encoder
        # Define the projection head
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_out_features_num, l2),
            nn.BatchNorm1d(l2),
            nn.ReLU(),
            nn.Linear(l2, l3)
        )

    def forward(self, x):
        encoder_representations = self.encoder(x)
        metric_embeddings = self.projection_head(encoder_representations)
        return metric_embeddings  # shape: (2 * batch_size, l3)

    def get_encoder(self):
        return self.encoder


# Create the train loop where the loss is printed every epoch for the simCLR model
def train_simCLR_model(data_loader, encoder, encoder_out_features_num, l2, l3, learning_rate, temperature):
    # Set up the model
    model = simCLRModel(encoder, encoder_out_features_num, l2, l3)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # the train loop
    loss_epochs = []
    for epoch in range(num_epochs):
        loss_step = []
        for x1, x2 in data_loader:
            x1, x2 = x1.to(device), x2.to(device)
            x = torch.cat([x1, x2], dim=0)
            optimizer.zero_grad()
            metric_embeddings = model(x)
            loss = nt_xent_loss(metric_embeddings, temperature)
            loss.backward()
            optimizer.step()
            loss_step.append(loss.item())
        loss_epochs.append(np.array(loss_step).mean())
        print(f"Epoch {epoch + 1}, Loss: {loss_epochs[epoch]:.4f}")

    return model, loss_epochs


# Define the encoder model
resnet_encoder = resnet18(weights=None)
resnet_encoder.fc = nn.Identity()  # remove classification head
resnet_encoder = resnet_encoder.to(device)

# Get the train and test data sets
trainset = torchvision.datasets.CIFAR10(root="./data",
                                        train=True,
                                        download=True,
                                        transform=None)
testset = torchvision.datasets.CIFAR10(root="./data",
                                       train=False,
                                       download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize(mean=test_mean_std[0],
                                                                                          std=test_mean_std[1])]))

transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.)),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean_std[0], std=train_mean_std[1])])

# Load the train set into the simCLRInput class
trainset = simCLRInput(trainset, transform)

# Load the data sets
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

trained_resnet_model, loss_epochs_resnet = train_simCLR_model(train_loader, resnet_encoder, 512, 128, 64, 1e-3, 0.5)

# Plot the loss
epochs = range(1, len(loss_epochs_resnet) + 1)

plt.figure(figsize=(10, 4))
plt.plot(epochs, loss_epochs_resnet)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Loss over Epochs')

# Save the trained model
torch.save(trained_resnet_model.state_dict(), "simCLR.pth")

# Save the plot
plt.savefig("simCLR.png", dpi=300, bbox_inches="tight")

"""Visualizing the separation of the features using a t-SNE plot"""

tsne_model = trained_resnet_model.get_encoder()
tsne_model.eval()
features = []
true_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = tsne_model(images)
        features.append(outputs)
        true_labels.append(labels)

# Concatenate the tensors of the batches as one tensor
features = torch.cat(features, dim=0)
true_labels = torch.cat(true_labels, dim=0)

# Draw the t-SNE plot
tsne = TSNE(n_components=2, random_state=30)
features_2d = tsne.fit_transform(features.cpu().numpy())

tsne_df = pd.DataFrame({
    "x": features_2d[:, 0],
    "y": features_2d[:, 1],
    "labels": true_labels.cpu().numpy()
})

# Define a map linking numeric labels to class names
class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
label_map = {i: name for i, name in enumerate(class_names)}

# Replace numeric predictions with class names in the DataFrame for plotting
tsne_df["labels"] = tsne_df["labels"].map(label_map)

# Plot t-SNE
plt.figure(figsize=(10, 8))

sns.scatterplot(
    data=tsne_df, x="x", y="y",
    s=10, hue="labels", palette="tab10", legend="full"
)

plt.title("t-SNE of features extracted by simCLR")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig("t-SNE after simCLR.png", dpi=300, bbox_inches="tight")

"""After running simCLR, I trained a linear classifier after the trained encoder (excluding the overhead projection 
preceding the NT-Xent loss) to test the efficiency of simCLR in extracting class-specific features from CIFAR10 images.
I chose to freeze the encoder rather than fine-tune it due to computational constraints."""

trainset = torchvision.datasets.CIFAR10(root="./data",
                                        train=True,
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize(mean=train_mean_std[0],
                                                                                           std=train_mean_std[1])]))
testset = torchvision.datasets.CIFAR10(root="./data",
                                       train=False,
                                       download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize(mean=test_mean_std[0],
                                                                                          std=test_mean_std[1])]))

# Load the data sets again
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


class LinearClassifier(nn.Module):
    def __init__(self, encoder, features_num, num_classes):
        super().__init__()
        self.encoder = encoder.get_encoder()
        self.classifier = nn.Linear(features_num, num_classes)

    def forward(self, x):
        with torch.no_grad():
            simCLR_features = self.encoder(x)
        return self.classifier(simCLR_features)


encoder = trained_resnet_model.get_encoder()
model = LinearClassifier(encoder, 512, 10)
model.to(device)
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

train_losses_epoch = []
val_losses_epoch = []

train_accuracy_epoch = []
val_accuracy_epoch = []

for epoch in range(num_epochs):
    train_losses_step = []
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses_step.append(loss.item())
    train_losses_epoch.append(np.array(train_losses_step).mean())

    model.eval()
    val_losses_step = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_losses_step.append(loss.item())
        val_losses_epoch.append(np.array(val_losses_step).mean())

    train_accuracy_epoch.append(get_accuracy(model, train_loader))
    val_accuracy_epoch.append(get_accuracy(model, test_loader))

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
plt.savefig("simCLR.png", dpi=300, bbox_inches="tight")

# Save the trained model
path = "./simCLR.pth"
torch.save(model.state_dict(), path)

# Checking the accuracy of a prediction for each class
classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# prepare to count predictions for each class
correct_pred = {class_name: 0 for class_name in classes}
total_pred = {class_name: 0 for class_name in classes}

with torch.no_grad():
    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for class_name, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[class_name]
    print(f"Accuracy for class: {class_name:5s} is {accuracy:.1f} %")
