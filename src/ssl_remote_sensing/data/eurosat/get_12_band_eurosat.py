from data.eurosat_dataset import (
    EuroSATDataset,
    InMemoryEuroSATDataset,
)
from torch.utils.data import DataLoader, random_split

def get_eurosat_dataloader(
    root,
    transform,
    batchsize,
    numworkers,
    split=False,
    in_memory: bool = False,
    max_samples: int = 21600,
):
    if in_memory:
        dataset = InMemoryEuroSATDataset(root, transform=transform)
    else:
        dataset = EuroSATDataset(root, transform=transform)
    print("[LOG] Total number of images: {}".format(len(dataset)))
    print(f"[LOG] Batch size is {batchsize}")

    if split:
        if max_samples != 21600:
            assert in_memory, "Fraction split only possible with in memory dataset"
            dataset = dataset.return_subset(n_total_samples=max_samples + 5400)
        train_set, val_set = random_split(dataset, [max_samples, 5400])
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
