class Hparams:
    def __init__(self):
        self.epochs = 10  # number of training epochs
        self.seed = 1234  # randomness seed
        self.cuda = False  # use nvidia gpu
        self.img_size = 64  # image shape
        self.save = "./saved_models/"  # save checkpoint
        self.gradient_accumulation_steps = 1  # gradient accumulation steps
        self.batch_size = 64
        self.lr = 1e-3
        self.embedding_size = 128  # papers value is 128
        self.temperature = 0.5  # 0.1 or 0.5
        self.weight_decay = 1e-6


def get_simclr_config():
    return Hparams()
