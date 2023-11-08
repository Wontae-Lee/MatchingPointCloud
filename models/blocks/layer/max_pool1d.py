import torch.nn as nn
import torch
from models.blocks.utils import return_distance


class MaxPool1D(nn.Module):
    def __init__(self, dgcnn_kernel_size=10):
        super().__init__()
        self.k = dgcnn_kernel_size

    def forward(self, src, tgt, src_coords, tgt_coords):
        src = self.max_pooling(src, src_coords)
        tgt = self.max_pooling(tgt, tgt_coords)

        return src, tgt

    def max_pooling(self, feats, coords):
        N, C = feats.size()

        # distance: [N, N]
        dist = return_distance(coords, coords)

        # index: [N, k]
        idx = dist.topk(k=self.k + 1, dim=-1, largest=False, sorted=True)[1]
        idx = idx[:, 1:]

        # neighbor features based on KNN index
        neighbor_feats = torch.gather(feats, dim=0, index=idx)

        # duplicate feats [N, C, k]
        feats = feats.unsqueeze(-1).repeat(1, 1, self.k)

        # duplicate neighbor's feats [N, C, k]
        neighbor_feats = neighbor_feats.unsqueeze(1).repeat(1, C, 1)

        # concatation
        feats_cat = torch.cat((feats, neighbor_feats - feats), dim=1)
        feats_cat = torch.max(feats_cat, dim=-1)[0]

        return feats_cat
