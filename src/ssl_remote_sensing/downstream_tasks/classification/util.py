import numpy as np
from torch.utils.data import SubsetRandomSampler


def get_subset_samplers_for_train_test_split(
    dataset_size: int, test_split_ratio: float = 0.2
):
    # Creating data indices for training and validation splits:
    random_seed = 42
    indices = list(range(dataset_size))
    split = int(np.floor(test_split_ratio * dataset_size))

    # Shuffle dataset
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    return train_sampler, test_sampler
