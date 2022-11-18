import torch

from ssl_remote_sensing.pretext_tasks.simclr.training import SimCLRTraining
from ssl_remote_sensing.models.ResNet18 import ResNetEncoder, resnet18_encoder
from ssl_remote_sensing.pretext_tasks.simclr.config import get_simclr_config
from ssl_remote_sensing.constants import RANDOM_INITIALIZATION


def load_encoder_checkpoint_from_pretext_model(
    path_to_checkpoint: str,
) -> ResNetEncoder:
    if "simclr" in path_to_checkpoint.lower():
        return (
            SimCLRTraining
            .load_from_checkpoint(path_to_checkpoint, config=get_simclr_config(), feat_dim=512)
            .model
            .backbone
        )
    elif "vae" in path_to_checkpoint.lower():
        raise NotImplementedError()
    elif "gan" in path_to_checkpoint.lower():
        raise NotImplementedError()
    elif RANDOM_INITIALIZATION in path_to_checkpoint:
        return resnet18_encoder()
    else:
        raise ValueError(
            f"Checkpoint name has to contain simclr, vae, or gan but was {path_to_checkpoint}"
        )
