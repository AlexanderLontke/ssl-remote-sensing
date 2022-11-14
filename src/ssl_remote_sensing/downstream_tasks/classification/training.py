from datetime import datetime

import torch
from torch import nn
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm


from torchvision.datasets import EuroSAT
from torch.utils.data import SubsetRandomSampler, DataLoader

from ssl_remote_sensing.downstream_tasks.classification.model import (
    DownstreamClassificationNet,
)

from ssl_remote_sensing.downstream_tasks.classification.util import (
    get_subset_samplers_for_train_test_split,
)


# Specify Data
eurosat_ds = EuroSAT(root="./", download=True, transform=T.ToTensor())

# Creating data indices for training and validation splits:
dataset_size = len(eurosat_ds)
test_split_ratio = 0.2

train_sampler, test_sampler = get_subset_samplers_for_train_test_split(
    dataset_size, test_split_ratio=test_split_ratio
)

train_dl = DataLoader(
    dataset=eurosat_ds,
    batch_size=128,
    sampler=train_sampler,
)

test_dl = DataLoader(
    dataset=eurosat_ds,
    batch_size=128,
    sampler=test_sampler,
)

# First of all, let's verify if a GPU is available on our compute machine. If not, the cpu will be used instead.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device used: {}".format(device))
model = DownstreamClassificationNet(input_dim=72 * 13 * 13).to(device)

# define the optimization criterion / loss function
loss_criterion = nn.CrossEntropyLoss().to(device)

# define learning rate and optimization strategy
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)


# specify the training parameters
num_epochs = 10  # number of training epochs
train_epoch_losses = []
validation_epoch_losses = []


for epoch in range(num_epochs):
    model.train()
    # init collection of mini-batch losses
    train_mini_batch_losses = []

    # iterate over all-mini batches
    for i, (images, labels) in tqdm(enumerate(train_dl), total=len(train_dl)):

        # push mini-batch data to computation device
        images = images.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        out = model(images)
        loss = loss_criterion(out, labels)
        loss.backward()
        # for p in model.parameters():
        #   print(p.grad)
        optimizer.step()

        # collect mini-batch reconstruction loss
        train_mini_batch_losses.append(loss.data.item())

    # determine mean min-batch loss of epoch
    train_epoch_loss = np.mean(train_mini_batch_losses)
    train_epoch_losses.append(train_epoch_loss)

    # Specify you are in evaluation mode
    model.eval()
    with torch.no_grad():
        validation_mini_batch_losses = []
        for (images, labels) in test_dl:
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            validation_epoch_loss = loss_criterion(outputs, labels)
            # collect mini-batch reconstruction loss
            validation_mini_batch_losses.append(loss.data.item())
        validation_epoch_loss = np.mean(validation_mini_batch_losses)
        validation_epoch_losses.append(validation_epoch_loss)

    # print epoch loss
    now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
    print(
        f"[LOG {now}] epoch: {epoch+1} train-loss: {train_epoch_loss} validation-loss: {validation_epoch_loss}"
    )
