import torch.nn as nn
import torch
from models.blocks.layer.max_pool1d import MaxPool1D
from models.blocks.layer.conv1d import Conv1d
from models.blocks.layer.instance_norm1d import InstanceNorm1d
from models.blocks.layer.leaky_relu import LeakyReLU


class GNN(nn.Module):

    def __init__(self, gnn_features_dimension, dgcnn_kernel_size):
        super().__init__()

        self.dgcnn_kernl_size = dgcnn_kernel_size

        # ********************
        #    Architecture
        # ********************

        self.max_pool_a = MaxPool1D(self.dgcnn_kernl_size)
        self.conv1d_a = Conv1d(gnn_features_dimension * 2, gnn_features_dimension, kernel_size=1, bias=False)
        self.conv1d_a_norm = InstanceNorm1d(gnn_features_dimension)
        self.leaky_relu_a = LeakyReLU(negative_slope=0.2)

        self.max_pool_b = MaxPool1D(self.dgcnn_kernl_size)
        self.conv1d_b = Conv1d(gnn_features_dimension * 2, gnn_features_dimension * 2, kernel_size=1, bias=False)
        self.conv1d_b_norm = InstanceNorm1d(gnn_features_dimension * 2)
        self.leaky_relu_b = LeakyReLU(negative_slope=0.2)

        self.conv1d_c = Conv1d(gnn_features_dimension * 4, gnn_features_dimension, kernel_size=1, bias=False)
        self.conv1d_c_norm = InstanceNorm1d(gnn_features_dimension)
        self.leaky_relu_c = LeakyReLU(negative_slope=0.2)

    def forward(self, src, tgt, src_coords, tgt_coords):
        # origin features
        src_0 = src
        tgt_0 = tgt

        # Edgeconv features
        src_1, tgt_1 = self.max_pool_a(src, tgt, src_coords, tgt_coords)
        src_1, tgt_1 = self.conv1d_a(src_1, tgt_1)
        src_1, tgt_1 = self.conv1d_a_norm(src_1, tgt_1)
        src_1, tgt_1 = self.leaky_relu_a(src_1, tgt_1)

        # Deeper layer from Edgeconv features
        src_2, tgt_2 = self.max_pool_b(src_1, tgt_1, src_coords, tgt_coords)
        src_2, tgt_2 = self.conv1d_b(src_2, tgt_2)
        src_2, tgt_2 = self.conv1d_b_norm(src_2, tgt_2)
        src_2, tgt_2 = self.leaky_relu_b(src_2, tgt_2)

        # output features
        src_3 = torch.cat((src_0, src_1, src_2), dim=1)
        tgt_3 = torch.cat((tgt_0, tgt_1, tgt_2), dim=1)
        src_3, tgt_3 = self.conv1d_c(src_3, tgt_3)
        src_3, tgt_3 = self.conv1d_c_norm(src_3, tgt_3)
        src_3, tgt_3 = self.leaky_relu_c(src_3, tgt_3)

        return src_3, tgt_3


if __name__ == "__main__":
    pass
