import torch.nn as nn
from models.blocks.layer.kpconv import KPConv
from models.blocks.layer.batch_norm import BatchNormBlock
from models.blocks.layer.leaky_relu import LeakyReLU


class Simple(nn.Module):

    def __init__(self, in_channels, out_channels, radius, name,
                 kp_extention=2.0, bn=True, bn_momentum=0.2):
        super(Simple, self).__init__()

        self.kpconv = KPConv(in_channels, out_channels, kp_extention, radius, name)
        self.batch_norm = BatchNormBlock(in_channel=out_channels // 2, bn=bn, bn_momentum=bn_momentum)
        self.leaky_relu = LeakyReLU()

        return

    def forward(self, src, tgt, src_coords, tgt_coords):
        # Execute Kernel point convolution layer
        src, tgt, src_coords, tgt_coords = self.kpconv(src, tgt, src_coords, tgt_coords)
        src, tgt = self.batch_norm(src, tgt)
        src, tgt = self.leaky_relu(src, tgt)
        return src, tgt, src_coords, tgt_coords


if __name__ == "__main__":
    pass
