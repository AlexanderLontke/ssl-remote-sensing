import os
import torch
import torch.nn as nn
import random
import wandb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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


def best_model_loader(pretext_model, saved_model_path):

    # restore pre-trained model snapshot
    best_model_name = saved_model_path

    # load state_dict from path
    state_dict_best = torch.load(best_model_name, map_location=torch.device("cpu"))

    # init pre-trained model class
    best_model = pretext_model

    # load pre-trained models
    best_model.load_state_dict(state_dict_best)

    return best_model


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
        return resnet18_encoder(channels = 12)

    else:
        raise ValueError(
            f"Checkpoint name has to contain simclr, vae, or bigan but was {path_to_checkpoint}"
        )


def get_metrics(true, preds):
    matrix = confusion_matrix(true.flatten(), preds.flatten())
    class_0, class_1 = matrix.diagonal() / matrix.sum(axis=1)
    print("***************** Metrics *****************")
    print("Class 0 (no water) accuracy: {:.3f}".format(class_0))
    print("Class 1 (water) accuracy: {:.3f}".format(class_1))
    print(
        "Overall accuracy: {:.3f}".format(
            accuracy_score(true.flatten(), preds.flatten())
        )
    )
    print("Equally Weighted accuracy: {:.3f}".format(0.5 * class_0 + 0.5 * class_1))
    print("IoU: {:.3f}".format(jaccard_score(true.flatten(), preds.flatten())))
    print("*******************************************")


def visualize_result(idx, bst_model, valset, device, wandb=wandb, model_name=None):

    if not idx:
        idx = random.randint(0, len(valset))
        print("Validation image ID: {}".format(idx))

    sample = valset.__getitem__(idx)
    img = sample["image"]
    label = sample["label"]

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))

    img_rgb = img[[3, 2, 1], :, :]
    img_rgb = np.transpose(img_rgb, (1, 2, 0))
    img_rgb = img_rgb / img_rgb.max()

    mask = label.squeeze()

    input_img = torch.from_numpy(img)
    input_img = torch.unsqueeze(input_img.float().to(device), 0)
    output = bst_model(input_img)
    output = torch.nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    output = output.to("cpu").squeeze(0).numpy()

    # wandb log
    img_log = wandb.Image(img_rgb, caption="Sentinel-2 RGB")
    wandb.log({f"Sentinal-2 RGB: {model_name}": img_log})
    mask_log = wandb.Image(
        Image.fromarray(np.uint8(mask)).convert("RGB"), caption="Groundtruth Mask"
    )
    wandb.log({f"Groundtruth Mask: {model_name}": mask_log})
    output_log = wandb.Image(
        Image.fromarray(np.uint8(output)).convert("RGB"), caption="Predicted Mask"
    )
    wandb.log({f"Predicted Mask: {model_name}": output_log})

    axs[0].imshow(img_rgb)
    axs[0].set_title("Sentinel-2 RGB")
    axs[0].axis("off")

    axs[1].imshow(mask)
    axs[1].set_title("Groundtruth Mask")
    axs[1].axis("off")

    axs[2].imshow(output)
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")

    plt.show()
