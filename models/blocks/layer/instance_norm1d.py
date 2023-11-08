import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class InstanceNorm1d(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.batch_norm = nn.InstanceNorm1d(in_channel)

    def __forward(self, feats):
        feats = feats.permute(1, 0)
        feats = self.batch_norm(feats)
        feats = feats.permute(1, 0)
        return feats

    def forward(self, src, tgt):
        # source features
        src = self.__forward(src)

        # target features
        tgt = self.__forward(tgt)

        return src, tgt
