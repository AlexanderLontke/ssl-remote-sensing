import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_remote_sensing.pretext_tasks.simclr.utils import device_as


class InfoNceLoss(nn.Module):
    """
    InfoNCE loss as in SimCLR paper
    """

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)
        ).float()

    @staticmethod
    def calc_similarity_batch(a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j as in the SimCLR paper
        """
        assert proj_1.shape[0] == proj_2.shape[0], "Projections' shapes need to match"
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(
            similarity_matrix / self.temperature
        )

        all_losses = torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss
