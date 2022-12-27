# Model Setup
class Hparams:
    def __init__(self):
        self.pretext_epochs = 30 # number of training epochs for pretext tasks
        self.seed = 42  # randomness seed
        # self.save = "./saved_models/"
        # self.gradient_accumulation_steps = 1  # gradient accumulation steps
        self.batch_size = 32
        self.lr = 1e-3
        # self.weight_decay = 1e-6
        self.latent_dim = 256
        self.optim = "Adam"
        # self.embedding_size = 128  # papers value is 128
        # self.temperature = 0.5  # 0.1 or 0.5
        self.cuda = True  # use coda
        self.transform = True
        self.split = False

def get_vae_config():
    return Hparams()
