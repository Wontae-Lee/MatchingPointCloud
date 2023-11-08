import torch.nn as nn
import torch
from models.blocks.utils import return_distance


class MatchLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.dist = nn.PairwiseDistance()
        self.eps = 1e-7
        self.soft_plus = nn.Softplus()

    def forward(self, src_coords, tgt_coords, radius=0.001):
        dist = return_distance(src_coords, tgt_coords)

        # idx: [N, 1]
        idx = dist.topk(k=2, dim=-1, largest=False, sorted=True)[1]
        idx = idx[:, 1].squeeze()

        match_bool = self.dist(src_coords, tgt_coords[idx]) < radius

        # closest features
        return self.soft_plus(torch.logsumexp(match_bool + self.eps, dim=0))


if __name__ == "__main__":
    pass
