import argparse
import torch
import os
import wandb
import torchvision.transforms as T
from pytorch_lightning.callbacks import GradientAccumulationScheduler, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from pretext_tasks.utils import reproducibility
from data.bigearthnet.bigearthnet_dataloader import get_bigearthnet_dataloader
from pretext_tasks.simclr.augmentation import Augment
from pretext_tasks.simclr.training import SimCLRTraining
from pretext_tasks.vae.model import VariationalAutoencoder
from pretext_tasks.gan.bigan import BIGAN
from pretext_tasks.simclr.config import get_simclr_config

parser = argparse.ArgumentParser(description='SSL Pretraining')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('--seed', type=int, default=1234, help='randomness seed')
parser.add_argument('--save', type=str, default='./saved_models/', help='save checkpoint')
# TODO: check if bigan uses RGB or multispectral
parser.add_argument('--in_channels', type=int, default=12, help='number of input channel')

parser.add_argument('--pretext', type=str, default='vae', help='pretext task')
parser.add_argument('--epochs', type=int, default=30, help='number of training epochs for pretext tasks')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer')
parser.add_argument('--img_size', type=int, default=64, help='image shape')

# vae and bigan special settings
parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension for vae model')

# simclr special settings
parser.add_argument('--temperature', type=float, default=0.5, help='temperature for simclr model')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay for simclr model')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--embedding_size', type=int, default=128, help='embedding size for simclr model')

# bigan special settings
parser.add_argument('--feature_maps_gen', type=int, default=64, help='feature maps for generator')
parser.add_argument('--feature_maps_disc', type=int, default=64, help='feature maps for discriminator')
parser.add_argument('--feature_maps_enc', type=int, default=64, help='feature maps for encoder')

# pretext_tasks = ['vae', 'simclr', 'bigan']

def main_pretext():

    # global pretext_tasks

    args = parser.parse_args()

    wandb.login()
    project_name = 'igarss-pretext'
    run_name = f'{args.pretext}-{args.batch_size}'
    wandb.init(project=project_name, name = run_name, config=args)

    wandb_logger = WandbLogger(
    project=project_name,
    config=args)

    available_gpus = torch.cuda.device_count()
    print("available_gpus:", available_gpus)
    file_name = f'{args.pretext}-batchsize_{args.batch_size}.ckpt'
    save_model_path = os.path.join(os.getcwd(), "saved_models/")
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    file_path = os.path.join(save_model_path, file_name)

    reproducibility(args)

    ########### DATA ###########

    if args.pretext == 'simclr':
        transform = Augment(args.img_size, normalizer=T.RandomCrop(64))
    if args.pretext == 'vae':
        transform = T.Compose([T.Resize([64,64], antialias=True)])
    if args.pretext == 'bigan':
       transform = T.Compose([T.Resize([64,64], antialias=True)])

    train_loader  = get_bigearthnet_dataloader(args.data, batch_size = args.batch_size, dataset_transform = transform, max_samples = 1000)
    # print("image shape: ", next(iter(train_loader))[0].shape)
    # assert next(iter(train_loader)).shape == (args.batch_size, 12, 64, 64)
# 
    ########### MODEL ###########

    if args.pretext == 'simclr':
        model = SimCLRTraining(
            config=args,
            feat_dim=512,
            )
    if args.pretext == 'vae':
        model =  VariationalAutoencoder(latent_dim = args.latent_dim, input_height=64, config = args)
    if args.pretext == 'bigan':
        model = BIGAN()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        model.cuda()

    ########### TRAINING ###########

    accumulator = GradientAccumulationScheduler(scheduling={0: 1})

    if args.pretext == 'simclr':
        checkpoint_callback = ModelCheckpoint(
            filename=file_name,
            dirpath=save_model_path,
            every_n_epochs=5,
            save_last=True,
            save_top_k=2,
            monitor="train/NTXentLoss",
            mode="min",
        )
        shared_trainer_kwargs = {
            "callbacks": [accumulator, checkpoint_callback],
            "max_epochs": args.epochs,
            "logger": wandb_logger,
            "log_every_n_steps": 1,
            "accelerator": "gpu",
        }
    if args.pretext == 'vae':
        checkpoint_callback = ModelCheckpoint(
            filename=file_name,
            dirpath=save_model_path,
            every_n_epochs=1,
            save_last=True,
            save_top_k=2,
            monitor="elbo",
            mode="min",
        )
        shared_trainer_kwargs = {
        "callbacks": [checkpoint_callback],
        "max_epochs": args.epochs,
        "logger": wandb_logger,
        "log_every_n_steps": 1,
        "accelerator": "gpu",
        }
    if args.pretext == 'bigan':
        checkpoint_callback = ModelCheckpoint(
            filename=file_name,
            dirpath=save_model_path,
            every_n_epochs=10,
            save_last=True,
            save_top_k=2,
            monitor="elb",
            mode="min",
        )
        shared_trainer_kwargs = {
        "callbacks": [checkpoint_callback],
        "max_epochs": args.epochs,
        "logger": wandb_logger,
        "log_every_n_steps": 1,
        "accelerator": "gpu",
        }

    trainer = Trainer(**shared_trainer_kwargs)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint(file_path)
    wandb.save(f"{run_name}.ckpt")
    wandb.finish()        

if __name__ == '__main__':
    main_pretext()
