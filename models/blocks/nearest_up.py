import torch
import torch.nn as nn
from models.blocks.layer.closest_pool1d import ClosestPool1D


class NearestUp(nn.Module):

    def __init__(self):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUp, self).__init__()
        self.closest_pool = ClosestPool1D()
        return

    def forward(self, src, tgt, src_coords, tgt_coords, shortcut_list: list):
        # featurs from shortcut
        src_shortcut, tgt_shortcut, src_shortcut_coords, tgt_shortcut_coords = shortcut_list

        # closest features
        src_closest, tgt_closest = self.closest_pool(src, tgt, src_coords, tgt_coords, src_shortcut_coords, tgt_shortcut_coords)

        # concatation
        src = torch.cat((src, src_closest), dim=0)
        tgt = torch.cat((tgt, tgt_closest), dim=0)

        return src, tgt, src_shortcut_coords, tgt_shortcut_coords
