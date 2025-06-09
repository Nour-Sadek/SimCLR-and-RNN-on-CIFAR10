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

# Training on the CIFAR-10 dataset using simCLR

Using simCLR involves applying a couple of augmentations on the input images, then running the augmented views through an 
encoder, in this case it is the ResNet18 model, and a non-linear Multi-Layer Perceptron (MLP) projection head to finally 
get evaluated by the NT-Xent Contrastive Loss Function. Due to limitations in computational resources, I opted to use 
ResNet18 instead of a bigger model like ResNet50 and arbitrarily chosen hyperparameters, and so the whole training data set 
was used rather than splitting it into training and validation, and the test set was used as the validation set instead. 
Nowhere in this implementation was the test set used to optimize the model.

### Choosing the transformations for the data augmentations

The paper tested a bunch of transformations and saw that doing random cropping of the images and strong color distortions 
gave the best results. Following what the paper outlines in its Appendix/Supplemental, these are types of augmentations 
that were performed on the input images:

    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean_std[0], std=train_mean_std[1])])

It needs to be mentioned that the types of data augmentations to would be effective would be highly dependent on the types 
of images being used. I used the same types of augmentations due to the assumed similarities between the CIFAR-10 and 
ImageNet images, however it might not be intuitively sound to try these same augmentations on different types of images, 
for example like medical or microscopy images, so different types of augmentations would need to be sampled.

### Implementation of the NT-Xent Loss Function

The NT-Xent loss function was implemented as outlined in the paper, where the output of the MLP __metric_embeddings__ 
is evaluated by this loss function scaled with the hyperparameter __temperature__. After defining the normalized and 
temperature scaled cosine similarity between the positive pairs in the numerator and all pairs, positive and negative, 
in the denominator, this fraction is fed through the Cross Entropy Loss function to output Normalized Temperature-Scaled 
Cross Entropy Loss (NT-Xent).

    def nt_xent_loss(metric_embeddings, temperature):
        double_batch_size = metric_embeddings.size(0)
        metric_embeddings = F.normalize(metric_embeddings, dim=1)

        # Create the numerator (cosine similarity between the sample and its positive)
        positives = torch.sum(metric_embeddings[:double_batch_size // 2] * metric_embeddings[double_batch_size // 2:],
                             dim=1) / temperature
        positives = torch.cat([positives, positives])  # for symmetry

        # Create the denominator (cosine similarity between the sample and its negatives and its positive)
        all_distances = torch.matmul(metric_embeddings, metric_embeddings.T) / temperature
        # Remove the cosine similarity between a sample and its self
        mask_self = ~torch.eye(double_batch_size, dtype=bool,
                               device=metric_embeddings.device)  # make an identity matrix that excludes the diagonal
        all_distances = all_distances[mask_self].view(double_batch_size, -1)  # remove the diagonal values, now it is of
                                                                              # size (double_batch_size, double_batch_size - 1)

        # Create the logits and labels as input for the cross_entropy function
        logits = torch.cat([positives.unsqueeze(1), all_distances], dim=1)
        labels = torch.zeros(double_batch_size, dtype=torch.long,
                             device=metric_embeddings.device)  # the first column (class 0) is where the positives are
        return F.cross_entropy(logits, labels)

### The Architecture of the simCLR model

The paper tested using no projection head as in the identity (as in feeding the output of the encoder to the Contrastive Loss function), 
a linear projection, or a non-linear projection, and saw that using a non-linear projection before evaluating with the loss 
function gave the best results. And so, a simple non-linear MLP was used with ReLU as the activation function. I also provide 
the opportunity to return the encoder only from the model, which will be beneficial later for dimensionality reduction analysis 
and applying a linear classifier.

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

### The Architecture of the Linear Classifier

After simCLR has trained the model to learn the representations, a simple linear classifier could be used to see how well 
these representations allow for the correct predictions of classifications. However, the paper showed that using the 
representations before the projection head (as in, the output representations from the encoder) gives better results, and so 
this simple linear classifier would accept the encoder (ResNet18 model) from simCLRModel as input.

    class LinearClassifier(nn.Module):
        def __init__(self, encoder, features_num, num_classes):
            super().__init__()
            self.encoder = encoder.get_encoder()
            self.classifier = nn.Linear(features_num, num_classes)

        def forward(self, x):
            with torch.no_grad():
                simCLR_features = self.encoder(x)
            return self.classifier(simCLR_features)

There are two ways this linear classifier could have been trained:
- Freeze the encoder and just train the linear classifier
- Fine-tune the encoder and allow its weights to be updated as well during the linear classification task.

For this task, I only opted for the first method to see how well applying simCLR on its own was able to create good 
representations and so, I specified that only the parameters of the classifier to be optimized. However, I hypothesize 
that trying to fine-tune the encoder during the linear classification task would give better classification results.

### Dimensionality Reduction Analysis

After representations of the training data were learned from simCLR and before running the linear classifier, the 
representations of the encoder were visualized using a t-SNE plot to see how well the model separated the test images 
based on their classifications.

### Results

The paper saw that using large batch sizes and training for longer epochs would lead to better representation learning, and 
or the CIFAR-10 dataset specifically, they used a batch size of 1024 and a temperature value of 0.5 and ran it for as 
many as 1000 epochs. Due to computational constraints, I only ran the simCLR model for 300 epochs with the same temperature 
value but an arbitrarily chosen learning rate and arbitrarily chosen sizes of the MLP layers.

This is how the loss of the training data set changed throughout training for 300 epochs.

![Image](https://github.com/user-attachments/assets/ac5233de-cb16-4226-a702-2ae83dbb48fa)

It seems like the model didn't converge and could have benefited from longer training time, and of course, resources 
permitting, might have given better results with optimized values for the learning rate and layers of the MLP.

After training, a linear classifier was trained on top of a frozen encoder to see how well the model learned the 
representations that would allow for a good classification prediction. Again, arbitrarily chosen learning rate was used.

This is how the loss and accuracy of the train and validation (test) data sets changed throughout training of the 
classifier for 100 epochs.

![Image](https://github.com/user-attachments/assets/a6bfb3f9-98d3-4ec3-b167-11d6f9f2c63f)

The model achieved a maximum validation accuracy of 73% which isn't that different from the training accuracy of about 
76% and the loss values of both the training and validation are pretty close to each other as well, so the model was able 
to successfully learn generalizable representations enough to prevent over-fitting of the data, but definitely as is 
indicated by the low training set accuracy, the model would benefit from better performance.

I also visualized the distribution of the learned representations using a t-SNE plot.

![Image](https://github.com/user-attachments/assets/5c24ccbb-e6ff-4deb-b325-7fce4c1e92f5)

It seems like the classes that belong to vehicles (car, truck, ship, plane) as well as the horse, deer and frog classes 
to some extent formed obvious clusters, however the remaining three classes dog, cat, and bird don't seem properly separated.

From this t-SNE plot, I expect that the accuracy of predictions for the dog, cat, and bird classes should be the lowest 
while the accuracy for the other classes which form obvious clusters and so the model has learned their representation 
relatively well should be comparatively much higher. To test this, rather than looking at the accuracy of the whole test 
dataset, I looked at the accuracy per class. This was the output, and indeed the accuracy of predictions for those three 
classes were the lowest, while the predictions for the vehicle classes were the highest.

    Accuracy for class: plane is 75.1 %
    Accuracy for class: car   is 82.8 %
    Accuracy for class: bird  is 59.1 %
    Accuracy for class: cat   is 47.0 %
    Accuracy for class: deer  is 66.3 %
    Accuracy for class: dog   is 61.4 %
    Accuracy for class: frog  is 75.3 %
    Accuracy for class: horse is 70.4 %
    Accuracy for class: ship  is 76.8 %
    Accuracy for class: truck is 77.5 %

### Possible Improvements

Multiple improvements could have been implemented to lead to better learned representations using this self-supervised 
contrastive loss method. Simple improvements include using a better performing encoder, such as ResNet50 instead of ResNet18, 
performing hyperparameter tuning for the MLP layer sizes, learning rate and its decay schedule, L2 regularization, and temperature, 
among others, as well as training for longer epochs and with larger batch sizes.

Also, during training of the linear classifier, the encoder could have also been fine-tuned alongside it rather just being 
frozen. Also, sampling different data augmentations could have been worthwhile, specifically looking at those augmentations 
that would induce appreciable changes to those images that the model failed to learn discernible features for like the images 
that belong to the dog, cat, and bird classes, which would require manual inspection of the training data.

# Repository Structure

This repository contains:

    rnn_hyperparameter_tune.py: Performing hyperparameter tuning using pytorch-lightning and Optuna to optimize the original 
    RNN model in rnn.py.

    rnn.py: Implementation of a Recurrent Neural Network where the loss and accuracy of the train and validation sets during 
    training are plotted.

    simCLR.py: Implementation of the simCLR algorithm, a simple framework for contrastive learning as well as performing 
    Dimensionality Reduction Analysis by plotting a t-SNE plot.

    functions.py: Three helper functions that are used across the python scripts.

    requirements.txt: List of required Python packages.

Python 3.12 version was used
