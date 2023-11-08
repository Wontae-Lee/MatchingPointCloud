import torch.nn as nn
import torch


class OverlapLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.dist = nn.PairwiseDistance()
        self.eps = 1e-7
        self.soft_plus = nn.Softplus()

    def forward(self, src_coords, tgt_coords):

        return torch.logsumexp(self.dist(src_coords, tgt_coords) + self.eps, dim=0)


if __name__ == "__main__":
    pass
