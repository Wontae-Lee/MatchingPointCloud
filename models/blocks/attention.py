import torch.nn as nn
import torch
from models.blocks.layer.linear import Linear
from models.blocks.layer.instance_norm1d import InstanceNorm1d
from models.blocks.layer.leaky_relu import LeakyReLU


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('dhn, dhm->hnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('hnm,dhm->dhn', prob, value), prob


class CrossAttention(nn.Module):

    def __init__(self, gnn_features_dimension, num_of_heads):
        super().__init__()

        # ********************
        #    Parameters
        # ********************
        assert gnn_features_dimension % num_of_heads == 0
        self.gnn_features_dimension = gnn_features_dimension
        self.dimension = gnn_features_dimension // num_of_heads
        self.num_of_heads = num_of_heads
        # ********************
        #    Architecture
        # ********************

        # Projection layers
        self.x_i_conv1d = nn.Conv1d(gnn_features_dimension, gnn_features_dimension, kernel_size=1)
        self.x_j1_conv1d = nn.Conv1d(gnn_features_dimension, gnn_features_dimension, kernel_size=1)
        self.x_j2_conv1d = nn.Conv1d(gnn_features_dimension, gnn_features_dimension, kernel_size=1)

        # Multi-layer perceptron
        self.mlp_input = Linear(gnn_features_dimension * 2, gnn_features_dimension * 2)
        self.mlp_output = Linear(gnn_features_dimension * 2, gnn_features_dimension)
        self.mlp_norm = InstanceNorm1d(gnn_features_dimension)
        self.mlp_relu = LeakyReLU()

        # Merge layer
        self.merge = nn.Conv1d(gnn_features_dimension, gnn_features_dimension, kernel_size=1)

    def _merge(self, feat_a, feat_b):
        """
        :param feat_a: [N, C]
        :param feat_b: [N, C]
        :return:
        """

        # feats.permute(1,0): change the axis to [C, N] because the rule of Pytorch library
        x_i_gnn = self.x_i_conv1d(feat_a.permute(1, 0))
        x_j1_gnn = self.x_j1_conv1d(feat_b.permute(1, 0))
        x_j2_gnn = self.x_j2_conv1d(feat_b.permute(1, 0))

        query = x_i_gnn.view(self.dimension, self.num_of_heads, -1)
        key = x_j1_gnn.view(self.dimension, self.num_of_heads, -1)
        value = x_j2_gnn.view(self.dimension, self.num_of_heads, -1)

        x, prob = attention(query, key, value)
        x = x.reshape(self.gnn_features_dimension, -1).contiguous()
        x = self.merge(x)

        # x.permute(1,0): to set the coordinates matrix[N, 3]
        return x.permute(1, 0), prob

    def forward(self, src, tgt, src_coords, tgt_coords):
        """
        :param tgt_coords:
        :param src_coords:
        :param src: [N, C]
        :param tgt: [N, C]
        :return:
        """
        src_cross, src_prob = self._merge(src, tgt)
        tgt_cross, tgt_prob = self._merge(tgt, src)

        src_cat = torch.cat((src_cross, src), dim=1)
        tgt_cat = torch.cat((tgt_cross, tgt), dim=1)

        src_cat, tgt_cat = self.mlp_input(src_cat, tgt_cat)
        src_cat, tgt_cat = self.mlp_output(src_cat, tgt_cat)
        src_cat, tgt_cat = self.mlp_norm(src_cat, tgt_cat)
        src_cat, tgt_cat = self.mlp_relu(src_cat, tgt_cat)

        src = src + src_cat
        tgt = tgt + tgt_cat

        return src, tgt, src_prob, tgt_prob, src_coords.clone().detach(), tgt_coords.clone().detach()
