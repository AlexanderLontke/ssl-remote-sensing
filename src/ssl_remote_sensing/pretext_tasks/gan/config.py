class Hparams:
    def __init__(self):
        self.epochs = 10  # number of training epochs
        self.seed = 1234  # randomness seed
        self.cuda = True  # use nvidia gpu
        self.img_size = 64  # image shape
        self.save = "./saved_models/"  # save checkpoint
        self.batch_size = 1000  # batch size
        self.lr = 0.0002  # learning rate
        self.latent_dim = 100  # latent dimension
        self.image_channels = 3  # number of image channels
        self.feature_maps_gen = 64
        self.feature_maps_disc = 64
        self.feature_maps_enc = 64


def get_bigan_config():
    return Hparams()
