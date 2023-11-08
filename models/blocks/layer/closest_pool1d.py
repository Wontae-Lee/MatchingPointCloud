import torch.nn as nn
from models.blocks.utils import return_distance


class ClosestPool1D(nn.Module):
    def __init__(self):
        super(ClosestPool1D, self).__init__()

    def forward(self, src, tgt, src_coords, tgt_coords, src_shortcut_coords, tgt_shortcut_coords):
        src = self.closest_pooling(src, src_coords, src_shortcut_coords)
        tgt = self.closest_pooling(tgt, tgt_coords, tgt_shortcut_coords)

        return src, tgt

    @staticmethod
    def closest_pooling(feats, coords, shortcut_coords):
        # distanc: [N, N]
        dist = return_distance(shortcut_coords, coords)

        # idx: [N, 1]
        idx = dist.topk(k=2, dim=-1, largest=False, sorted=True)[1]
        idx = idx[:, 1]

        return feats[idx]
