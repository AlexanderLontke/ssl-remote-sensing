# import torchvision
# import torchvision.transforms as transforms
# import torch

# classes = (
#     "AnnualCrop",
#     "Forest",
#     "HerbaceousVegetation",
#     "Highway",
#     "Industrial",
#     "Pasture",
#     "PermanentCrop",
#     "Residential",
#     "River",
#     "SeaLake",
# )


# def EurosatDataloader(batch_size):
#     transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#         ]
#     )
#     trainset = torchvision.datasets.EuroSAT(
#         root="./data", download=True, transform=transform
#     )
#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, shuffle=True
#     )
#     print("[LOG] Size of the image is: {}".format(trainset[0][0].shape))
#     print("[LOG] Total number of batches in the dataloader: %d" % len(trainloader))
#     return trainloader
