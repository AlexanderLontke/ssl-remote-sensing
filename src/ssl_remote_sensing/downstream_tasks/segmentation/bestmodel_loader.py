import os
import torch


def best_model_loader(pretext_model, pretext_model_dir, pretext_model_path):

    # restore pre-trained model snapshot
    best_model_name = os.path.join(pretext_model_dir, pretext_model_path)

    # load state_dict from path
    state_dict_best = torch.load(best_model_name, map_location=torch.device("cpu"))

    # init pre-trained model class
    best_model = pretext_model

    # load pre-trained models
    best_model.load_state_dict(state_dict_best)
