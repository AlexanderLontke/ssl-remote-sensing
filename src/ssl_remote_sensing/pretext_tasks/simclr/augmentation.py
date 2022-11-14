import torchvision.transforms as T


class Augment:
    """
    a probabilistic data augmentation module
    Transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x_i and  x_j which we consider a positive pair.
    """

    def __init__(self, img_size, norm_means, norm_stds, s=1):
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        # 10% of the image
        blur = T.GaussianBlur(
            kernel_size=(
                3,
                3,
            ),
            sigma=(0.1, 0.2),
        )

        self.train_transform = T.Compose(
            [
                # Crop image on a random scale from 7% tpo 100%
                # Only Use cropping since according to Wang et al. 2022
                # it is the only augmentation that has a significant impact
                T.RandomResizedCrop(size=img_size),
                # # Flip image horizontally with 50% probability
                # T.RandomHorizontalFlip(p=0.5),
                # # Apply heavy color jitter with 80% probability
                # T.RandomApply([color_jitter], p=0.8),
                # # Apply gaussian blur with 50% probability
                # T.RandomApply([blur], p=0.5),
                # # Convert RGB images to grayscale with 20% probability
                # T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(
                    mean=norm_means,
                    std=norm_stds,
                ),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
