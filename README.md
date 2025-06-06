# Implementing Self-Supervised Contrastive Learning using simCLR as well as separately implementing an RNN on the CIFAR-10 dataset

After learning about RNNs and LSTMs as well as about Contrastive Learning, and specifically simCLR from the paper "A Simple Framework for 
Contrastive Learning of Visual Representations", I practiced implementing a simple RNN and the simCLR model using PyTorch.

# Training on the CIFAR-10 dataset using an RNN and an LSTM

Recurrent Neural Networks (RNNs) are usually used on sequential data and not on simple image classification tasks. However, 
the objective here was to learn how to implement an RNN in PyTorch and how to feed in sequential data. A good validation 
accuracy is not expected to be obtained.

### How was sequential data generated from the images

CIFAR-10 images are 3x32x32. I chose to let the inputs be each row of those images, where that row is of size 3*32 (32 
pixels from the Red channel, followed by 32 from the Blue, then 32 from the Green), and so each image would pass 32 
inputs (32 total rows) into the network where each input is of size 96.

This is how each image in a batch is transformed:

    for images, labels in train_loader: # images: (batch_size, 3, 32, 32)
        # Reshape images to (batch_size, 32, 96) where the corresponding rows from each channel are concatenated as one row
        images = images.permute(0, 2, 3, 1)  # to (batch_size, rows, columns, channels)
        images = images.reshape(images.shape[0], images.shape[1], -1)

### The architecture of the RNN

Made a simple RNN model whose output is passed through a non-linear projection head that outputs a score for 
each of the 10 possible classes.

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

PyTorch-Lightning and Optuna were used to tune the following hyperparameters:

    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    num_layers = trial.suggest_categorical("num_layers", [1, 3, 5])
    l2 = trial.suggest_categorical("l2", [2 ** i for i in range(4, 7)])
    l3 = trial.suggest_categorical("l3", [2 ** i for i in range(4, 7)])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 5e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2 ** i for i in range(5, 9)])

After running the optimizer for 100 trials for a maximum of 10 epochs per trial, these were the hyperparameters values that 
produced the lowest validation error, which were used during training:

    {'hidden_size': 256, 'num_layers': 1, 'l2': 64, 'l3': 64, 'learning_rate': 0.00021990370923287235, 
    'weight_decay': 0.0005182777790955891, 'batch_size': 64}

After that, the model was trained for a total of 100 epochs were Cross Entropy was used as the loss function, Adam as the 
optimizer with default beta values, and a scheduler that decreases the learning rate with patience of 10 by a factor of 0.1 
based on no improvements seen on minimizing the validation loss.

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00022, weight_decay=0.00052)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

### Results

This is how the loss and accuracy of the train and validation data sets changed throughout training for 100 
epochs when the Multi-Layer Perceptron (MLP) was preceded by an RNN

![Image](https://github.com/user-attachments/assets/88cc453f-aac9-4378-af2d-a8117643e9d6)

Using a simple RNN only lead to getting a plateaued accuracy of about 50% for the validation set, and while in the 
beginning of training it didn't seem like there was much overfitting going on, as the model trained for more epochs, 
it seemed like it was able to increase its performance on the training set by overfitting to it.

Using LSTMs usually leads to better results, so I opted to replace the RNN modules with LSTM modules to see how different 
the performance will be. It was a simple function call change, and these are the reuslts:

![Image](https://github.com/user-attachments/assets/a9bd3170-6b8d-4950-bd9f-6950e734fc32)

Even though the model was trained for the same number of epochs, using an LSTM lead to a significant increase in the model's 
performance in both the training and validation sets, however, at around 50 epochs the model's performance reached a plateau, 
and similarly to when RNNs were used, it greatly overfit to the training data.

Again, since it is a simple function call change, I wanted to see how different will the performance be when a Gated Recurrent Unit (GRU) 
is used. Since it is similar to an LSTm but more computationally efficient, I wouldn't expect much difference in the results.

![Image](https://github.com/user-attachments/assets/5481615d-e3c8-4acf-bc1c-b44be085d568)

As expected, even though it lead to improvement in validation loss, the changes in training and validation accuracy followed 
the same pattern as when as LSTM was used.
