import os
import torch
import torch.nn as nn
from ssl_remote_sensing.pretext_tasks.vae.model import VariationalAutoencoder
from ssl_remote_sensing.pretext_tasks.simclr.training import SimCLRTraining
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score

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


def display_outputs(idx=None, multi=False):
    # Pick a random index if none is specified
    if not idx:
        idx = random.randint(0, len(valset))
    print('Validation image ID: {}'.format(idx))
    
    # Get Sentinel 2 and Sentinel 1 data
    s2_data = torch.unsqueeze(valset.__getitem__(idx)['s2_img'].float().to(device), 0)
    s1_data = torch.unsqueeze(valset.__getitem__(idx)['s1_img'].float().to(device), 0)
    
    # Get predictions from the model
    if multi:
        output = model(s1_data, s2_data)
    else:
        output = model(s2_data)
    
    # Threshold the output to generate the binary map (FYI: the threshold value "0" can be tuned as any other hyperparameter)
    output_binary = torch.zeros(output.shape)
    output_binary[output >= 0] = 1
    
    get_metrics(valset.__getitem__(idx)['mask'], output_binary)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    axes[0].imshow(np.transpose(valset.__getitem__(idx)['s2_img'][[3,2,1],:,:], (1, 2, 0)) / valset.__getitem__(idx)['s2_img'].max())
    axes[0].set_title('True Color Sentinel-2')
    axes[2].imshow(valset.__getitem__(idx)['mask'], cmap='Blues')
    axes[2].set_title('Groundtruth')
    axes[1].imshow(output_binary.squeeze(), cmap='Blues')
    axes[1].set_title('Predicted Mask')