import torch
import numpy as np

import matplotlib.pyplot as plt
import torchvision.transforms as T


def imshow(img, norm_means, norm_stds):
    """
    shows an imagenet-normalized image on the screen
    """
    mean = torch.tensor(norm_means, dtype=torch.float32)
    std = torch.tensor(norm_stds, dtype=torch.float32)
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    npimg = unnormalize(img).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def reproducibility(config):
    SEED = int(config.seed)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if config.cuda:
        torch.cuda.manual_seed(SEED)


def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)


def define_parameter_groups(model, weight_decay, optimizer_name):
    def exclude_from_weight_decay_and_adaptation(name):
        if "bn" in name:
            return True
        if optimizer_name == "lars" and "bias" in name:
            return True

    param_groups = [
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if not exclude_from_weight_decay_and_adaptation(name)
            ],
            "weight_decay": weight_decay,
            "layer_adaptation": True,
        },
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if exclude_from_weight_decay_and_adaptation(name)
            ],
            "weight_decay": 0.0,
            "layer_adaptation": False,
        },
    ]
    return param_groups
