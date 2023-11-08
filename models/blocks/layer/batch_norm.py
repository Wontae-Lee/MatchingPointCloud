import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BatchNormBlock(nn.Module):

    def __init__(self, in_channel, bn, bn_momentum):

        super(BatchNormBlock, self).__init__()
        self.in_channel = in_channel
        self.bn = bn
        self.bn_momentum = bn_momentum

        if bn:
            # self.batch_norm = nn.BatchNorm1d(in_dim, momentum = bn_momentum)
            self.batch_norm = nn.InstanceNorm1d(in_channel, momentum=bn_momentum)
        else:
            # 가중치 0으로 초기화
            self.bias = Parameter(torch.zeros(in_channel, dtype=torch.float32), requires_grad=True)
        return

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
