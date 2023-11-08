import torch.nn as nn
import torch


class DistanceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.dist = nn.PairwiseDistance()
        self.eps = 1e-7

    def forward(self, src, tgt, src_coords, tgt_coords):
        return torch.sum(self.dist(src, tgt) + self.eps, dim=0)


if __name__ == "__main__":
    pass
