import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate the appropriate transform depending on data set, as well as its mean and std
def apply_transforms(mean_std_tuple: tuple[torch.tensor, torch.tensor],
                     train: bool) -> transforms:
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std_tuple[0], std=mean_std_tuple[1])])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std_tuple[0], std=mean_std_tuple[1])])


# Determine the accuracy of a model's predictions on a specific data set
def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images = images.permute(0, 2, 3, 1)
            images = images.reshape(images.shape[0], images.shape[1], -1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    accuracy = round((correct / total) * 100, 2)
    return accuracy


# Define the NT-Xent loss according to the simCLR paper
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
