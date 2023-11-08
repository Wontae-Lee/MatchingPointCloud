from models.blocks.layer.kpconv import KPConv
from models.blocks.layer.conv1d import Conv1d
from models.blocks.layer.batch_norm import BatchNormBlock
from models.blocks.layer.leaky_relu import LeakyReLU
import torch.nn as nn


class ResnetA(nn.Module):

    def __init__(self, in_channels, out_channels, radius, name,
                 kp_extent=2.0,
                 conv_radius=2.5,
                 bn=True,
                 bn_momentum=0.2):
        super(ResnetA, self).__init__()

        # output from shortcut
        shortcut_out_channels = in_channels // 4

        # get kp_extent from current radius
        current_extent = radius * kp_extent / conv_radius
        # save parameters
        self.radius = radius

        '''
        Neural Network
        '''
        # Adding layer
        self.conv1d = Conv1d(in_channels, out_channels, kernel_size=1)

        # shortcut layer
        self.conv1d_in = Conv1d(in_channels, shortcut_out_channels, kernel_size=1)
        self.KPConv = KPConv(in_channels=shortcut_out_channels, out_channels=shortcut_out_channels, kp_extention=current_extent, radius=radius, name=name)
        self.conv1d_out = Conv1d(in_channels=shortcut_out_channels, out_channels=out_channels, kernel_size=1)
        self.batch_norm = BatchNormBlock(in_channel=out_channels // 2, bn=bn, bn_momentum=bn_momentum)
        # Other operations
        self.leaky_relu = LeakyReLU()

        return

    # def __save_features(self, src, tgt):
    #     self.src_for_add = src.clone().detach()
    #     self.tgt_for_add = tgt.clone().detach()
    #     self.src_for_add, self.tgt_for_add = self.conv1d(self.src_for_add, self.tgt_for_add)
    #     print(self.src_for_add.shape)
    #
    # def __add_features(self, src, tgt):
    #     print(src.shape)
    #     src += self.src_for_add
    #     tgt += self.tgt_for_add

    def forward(self, src, tgt, src_coords, tgt_coords):
        # save features before shortcut
        # self.__save_features(src, tgt)
        src, tgt = self.conv1d_in(src, tgt)
        src, tgt, src_coords, tgt_coords = self.KPConv(src, tgt, src_coords, tgt_coords)
        src, tgt = self.conv1d_out(src, tgt)

        # add features after shortcut
        # self.__add_features(src, tgt)

        # normalization
        src, tgt = self.batch_norm(src, tgt)
        src, tgt = self.leaky_relu(src, tgt)

        return src, tgt, src_coords, tgt_coords


class ResnetB(nn.Module):

    def __init__(self, in_channels, out_channels, radius, name,
                 kp_extent=2.0,
                 conv_radius=2.5,
                 bn=True,
                 bn_momentum=0.2):
        super(ResnetB, self).__init__()

        # output from shortcut
        shortcut_out_channels = in_channels // 2

        # get kp_extent from current radius
        current_extent = radius * kp_extent / conv_radius

        # save parameters
        self.radius = radius

        '''
        Neural Network
        '''

        self.conv1d_in = Conv1d(in_channels, shortcut_out_channels, kernel_size=1)
        self.KPConv = KPConv(in_channels=shortcut_out_channels, out_channels=shortcut_out_channels, kp_extention=current_extent, radius=radius, name=name)
        self.conv1d_out = Conv1d(in_channels=shortcut_out_channels, out_channels=out_channels, kernel_size=1)

        self.batch_norm = BatchNormBlock(in_channel=out_channels // 2, bn=bn, bn_momentum=bn_momentum)
        # Other operations
        self.leaky_relu = LeakyReLU()

        return

    def forward(self, src, tgt, src_coords, tgt_coords):
        src, tgt = self.conv1d_in(src, tgt)

        src, tgt, src_coords, tgt_coords = self.KPConv(src, tgt, src_coords, tgt_coords)
        src, tgt = self.conv1d_out(src, tgt)
        src, tgt = self.batch_norm(src, tgt)
        src, tgt = self.leaky_relu(src, tgt)

        return src, tgt, src_coords, tgt_coords
