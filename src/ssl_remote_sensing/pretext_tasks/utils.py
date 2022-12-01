import torch

from ssl_remote_sensing.pretext_tasks.simclr.training import SimCLRTraining
from ssl_remote_sensing.models.ResNet18 import ResNetEncoder, resnet18_encoder
from ssl_remote_sensing.pretext_tasks.simclr.config import get_simclr_config
from ssl_remote_sensing.pretext_tasks.vae.config import get_vae_config
from ssl_remote_sensing.constants import RANDOM_INITIALIZATION
from ssl_remote_sensing.pretext_tasks.vae.model import VariationalAutoencoder
from ssl_remote_sensing.pretext_tasks.gan.bigan_encoder import BiganResnetEncoder
from ssl_remote_sensing.models.ResNet18 import resnet18_basenet
from ssl_remote_sensing.pretext_tasks.gan.config import get_bigan_config


def load_encoder_checkpoint_from_pretext_model(
    path_to_checkpoint: str,
) -> ResNetEncoder:
    if "simclr" in path_to_checkpoint.lower():
        return SimCLRTraining.load_from_checkpoint(
            path_to_checkpoint, config=get_simclr_config(), feat_dim=512
        ).model.backbone
    elif "vae" in path_to_checkpoint.lower():

        best_model = VariationalAutoencoder(
            latent_dim=get_vae_config().latent_dim, config=get_vae_config()
        )
        state_dict_best = torch.load(
            path_to_checkpoint, map_location=torch.device("cpu")
        )
        best_model.load_state_dict(state_dict_best)
        return best_model.encoder

    elif "bigan" in path_to_checkpoint.lower():
        resnet_basemodel = resnet18_basenet(False)
        config = get_bigan_config()
        model = BiganResnetEncoder(
            config.latent_dim,
            config.feature_maps_enc,
            config.image_channels,
            pretrained_model=resnet_basemodel,
        )
        state_dict_best = torch.load(
            path_to_checkpoint, map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict_best)
        return model
    elif (
        path_to_checkpoint == "/content/drive/MyDrive/deep_learning_checkpoints/random"
    ):
        return resnet18_encoder()
    else:
        raise ValueError(
            f"Checkpoint name has to contain simclr, vae, or bigan but was {path_to_checkpoint}"
        )
