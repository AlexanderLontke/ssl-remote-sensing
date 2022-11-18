import torch

from ssl_remote_sensing.pretext_tasks.simclr.resnet_18_backbone import AddProjection
from ssl_remote_sensing.models.ResNet18 import ResNetEncoder, resnet18_encoder
from ssl_remote_sensing.pretext_tasks.simclr.config import get_simclr_config
from ssl_remote_sensing.constants import RANDOM_INITIALIZATION, STATE_DICT_KEY


def load_encoder_checkpoint_from_pretext_model(
    path_to_checkpoint: str,
) -> ResNetEncoder:
    checkpoint = torch.load(path_to_checkpoint)
    if hasattr(checkpoint, STATE_DICT_KEY):
        checkpoint = checkpoint[STATE_DICT_KEY]
    if "simclr" in path_to_checkpoint.lower():
        return (
            AddProjection(config=get_simclr_config())
            .load_state_dict(checkpoint)
            .backbone
        )
    elif "vae" in path_to_checkpoint.lower():
        raise NotImplementedError()
    elif "gan" in path_to_checkpoint.lower():
        raise NotImplementedError()
    elif path_to_checkpoint == RANDOM_INITIALIZATION:
        return resnet18_encoder()
    else:
        raise ValueError(
            f"Checkpoint name has to contain simclr, vae, or gan but was {path_to_checkpoint}"
        )
