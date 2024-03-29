import torch

# from pretext_tasks.simclr.training import SimCLRTraining
from models.ResNet18 import ResNetEncoder, resnet18_encoder, resnet18_basenet
from pretext_tasks.simclr.config import get_simclr_config
from pretext_tasks.simclr.training import SimCLRTraining
from pretext_tasks.vae.config import get_vae_config
from pretext_tasks.gan.config import get_bigan_config
from constants import RANDOM_INITIALIZATION
from pretext_tasks.vae.model import VariationalAutoencoder
from pretext_tasks.gan.bigan_encoder import BiganResnetEncoder

import numpy as np


def reproducibility(config):
    SEED = int(config.seed)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

def load_encoder_checkpoint_from_pretext_model(
    path_to_checkpoint: str,
) -> ResNetEncoder:
    if "simclr" in path_to_checkpoint.lower():
        return SimCLRTraining.load_from_checkpoint(
            path_to_checkpoint, config=get_simclr_config(), feat_dim=512
        ).model.backbone
    elif "vae" in path_to_checkpoint.lower():

        # return VariationalAutoencoder.load_from_checkpoint(
        #    path_to_checkpoint,latent_dim =256,input_height=64, config= get_vae_config()).encoder

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
    elif "random" in path_to_checkpoint:
        return resnet18_encoder(channels = 12) 
    else:
        raise ValueError(
            f"Checkpoint name has to contain simclr, vae, or bigan but was {path_to_checkpoint}"
        )
