from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as T

means = [0.3444, 0.3803, 0.4078]
stds = [0.2037, 0.1366, 0.1148]

# if train_config.transform:

#     transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean = means,
#                     std = stds) # params computed from the eurosat data
#                 ])
# else:
#     transform = transforms.Compose([
#                   transforms.ToTensor(),
#                   ])


def get_eurosat_normalizer():
    return T.Normalize(mean=means, std=stds)


def get_eurosat_dataloader(root, transform, batchsize, numworkers, split=False):

    dataset = EuroSAT(root, transform=transform, download=True)
    print("[LOG] Total number of images: {}".format(len(dataset)))
    print("[LOG] Size of the image is: {}".format(dataset[0][0].shape))
    print(f"[LOG] Batch size is {batchsize}")

    if split:

        train_set, val_set = random_split(dataset, [21600, 5400])
        print(f"[LOG] Total images in the train set is: {len(train_set)}")

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batchsize,
            num_workers=numworkers,
            shuffle=True,
        )
        print(
            "[LOG] Total number of batches in the trainloader: %d" % len(train_loader)
        )

        val_loader = DataLoader(
            dataset=val_set, batch_size=batchsize, num_workers=numworkers, shuffle=True
        )
        print("[LOG] Total number of batches in the valloader: %d" % len(val_loader))
        return train_loader, val_loader

    else:
        train_set = dataset
        print(f"[LOG] Total images in the train set is: {len(train_set)}")

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batchsize,
            num_workers=numworkers,
            shuffle=True,
        )
        print(
            "[LOG] Total number of batches in the trainloader: %d" % len(train_loader)
        )
        return train_loader


# train_loader = get_eurosat("./",transform = transform, batchsize = 16, numworkers = cpu_count(), split = False)
