from ssl_remote_sensing.data.eurosat.eurosat_dataset import EuroSATDataset
from torch.utils.data import DataLoader, random_split


def get_eurosat_dataloader(root, transform, batchsize, numworkers, split=False):
    dataset = EuroSATDataset(root, transform=transform)
    print("[LOG] Total number of images: {}".format(len(dataset)))
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
