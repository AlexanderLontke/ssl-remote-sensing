# Model Setup
class Hparams:
    def __init__(self):
        self.pretext_epochs = 30 # number of training epochs for pretext tasks
        self.seed = 42  # randomness seed
        self.batch_size = 32
        self.lr = 1e-3
        self.latent_dim = 256
        self.optim = "Adam"
        self.cuda = True  # use coda
        self.transform = True
        self.split = False

def get_vae_config():
    return Hparams()
