import os
import torch
import torch.nn as nn
from ssl_remote_sensing.pretext_tasks.vae.model import VariationalAutoencoder
from ssl_remote_sensing.pretext_tasks.simclr.training import SimCLRTraining
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score
from ssl_remote_sensing.pretext_tasks.simclr.training import SimCLRTraining
from ssl_remote_sensing.models.ResNet18 import ResNetEncoder, resnet18_encoder
from ssl_remote_sensing.pretext_tasks.simclr.config import get_simclr_config
from ssl_remote_sensing.pretext_tasks.vae.config import get_vae_config
from ssl_remote_sensing.pretext_tasks.vae.model import VariationalAutoencoder
from ssl_remote_sensing.pretext_tasks.gan.bigan_encoder import BiganResnetEncoder
from ssl_remote_sensing.models.ResNet18 import resnet18_basenet
from ssl_remote_sensing.pretext_tasks.gan.config import get_bigan_config

# from ssl_remote_sensing.pretext_tasks.simclr.config import get_simclr_config


def load_best_model(model_name, model_save_name, model_dir, train_config=None):

    if model_name == "VAE":
        best_model = VariationalAutoencoder(
            latent_dim=train_config.latent_dim, config=train_config
        )

    elif model_name == "GAN":
        raise NotImplementedError()

    elif model_name == "SimCLR":
        raise NotImplementedError()
        best_model = SimCLRTraining(config=train_config, feat_dim=512)

    # load state_dict from path
    best_model_path = os.path.join(model_dir, model_save_name)
    print(best_model_path)
    state_dict_best = torch.load(best_model_path, map_location=torch.device("cpu"))
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

    new_weight = torch.Tensor(
        module.out_channels, new_in_channels // module.groups, *module.kernel_size
    )
    for i in range(new_in_channels):
        new_weight[:, i] = weight[:, i % default_in_channels]

    new_weight = new_weight * (default_in_channels / new_in_channels)
    module.weight = nn.parameter.Parameter(new_weight)

    # make sure in_channel is changed
    assert module.in_channels == new_in_channels


def get_metrics(true, preds):
    matrix = confusion_matrix(true.flatten(), preds.flatten())
    class_0, class_1 = matrix.diagonal() / matrix.sum(axis=1)
    print('***************** Metrics *****************')
    print('Class 0 (no water) accuracy: {:.3f}'.format(class_0))
    print('Class 1 (water) accuracy: {:.3f}'.format(class_1))
    print('Overall accuracy: {:.3f}'.format(accuracy_score(true.flatten(), preds.flatten())))
    print('Equally Weighted accuracy: {:.3f}'.format(0.5 * class_0 + 0.5 * class_1))
    print('IoU: {:.3f}'.format(jaccard_score(true.flatten(), preds.flatten())))
    print('*******************************************')

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
    
        best_model = VariationalAutoencoder(latent_dim = get_vae_config().latent_dim, config = get_vae_config())
        state_dict_best = torch.load(path_to_checkpoint, map_location=torch.device('cpu'))
        best_model.load_state_dict(state_dict_best)
        return best_model.encoder

    elif "gan" in path_to_checkpoint.lower():
        resnet_basemodel = resnet18_basenet(False)
        config = get_bigan_config()
        model = BiganResnetEncoder(config.latent_dim, config.feature_maps_enc, config.image_channels, pretrained_model=resnet_basemodel)
        state_dict_best = torch.load(path_to_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict_best)
        return model
   
    elif path_to_checkpoint == '/content/drive/MyDrive/deep_learning_checkpoints/random':
        return resnet18_encoder()

    else:
        raise ValueError(
            f"Checkpoint name has to contain simclr, vae, or gan but was {path_to_checkpoint}"
        )