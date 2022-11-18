import os
import torch
import torch.nn as nn
from ssl_remote_sensing.pretext_tasks.vae.model import VariationalAutoencoder
from ssl_remote_sensing.pretext_tasks.simclr.training import SimCLRTraining
# from ssl_remote_sensing.pretext_tasks.simclr.config import get_simclr_config
from ssl_remote_sensing.pretext_tasks.simclr.resnet_18_backbone import AddProjection

def load_best_model(model_name, model_save_name, model_dir, train_config = None):

    if model_name == "VAE":
        best_model = VariationalAutoencoder(latent_dim = train_config.latent_dim, config = train_config)

    elif model_name == "GAN":
        raise NotImplementedError()

    elif model_name == "SimCLR":
        raise NotImplementedError()
        best_model = SimCLRTraining(config=train_config,feat_dim=512)

    # load state_dict from path
    best_model_path = os.path.join(model_dir, model_save_name)
    print(best_model_path)
    state_dict_best = torch.load(best_model_path, map_location=torch.device('cpu'))
    # load pre-trained models
    best_model.load_state_dict(state_dict_best)

    return best_model


# funciton to change first convolution layer input channels => make random kaiming normal initialization

def patch_first_conv(encoder, new_in_channels, default_in_channels=3):

    for module in encoder.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            print("Module to be convoluted: ", module)
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels  
    print("New module: ", module)       

    new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
    for i in range(new_in_channels):
        new_weight[:, i] = weight[:, i % default_in_channels]

    new_weight = new_weight * (default_in_channels / new_in_channels)
    module.weight = nn.parameter.Parameter(new_weight)

    # make sure in_channel is changed
    assert module.in_channels == new_in_channels
