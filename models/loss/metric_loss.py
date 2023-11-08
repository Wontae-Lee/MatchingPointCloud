import torch.nn as nn
import torch
from models.loss.circle_loss import CircleLoss
from models.loss.overlap_loss import OverlapLoss
from models.loss.match_loss import MatchLoss
from torch.nn.parameter import Parameter
from models.blocks.utils import return_distance

#
# def corresponding(src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords):
#     # Create probobility matrix [N X M] from the different workflow
#     src_prob = torch.sum(src_prob.permute(1, 2, 0), dim=2)
#     tgt_prob = torch.sum(tgt_prob.permute(2, 1, 0), dim=2)
#
#     # Get index of points that are assumed to be in the same positions which is obtained
#     # from the source file and the target file, respectively
#     src_idx = src_prob.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)
#     tgt_idx = tgt_prob.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)
#
#     # matching coordinates
#     src_match_coords = src_gnn_coords[src_idx == tgt_idx]
#     tgt_match_coords = tgt_gnn_coords[src_idx[src_idx == tgt_idx]]
#
#     dist_positive = return_distance(src_match_coords, src_coords)
#     dist_negative = return_distance(src_gnn_coords[src_idx != tgt_idx], src_coords)
#
#     idx_positive = dist_positive.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)
#     idx_negative = dist_negative.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)
#
#     return idx_positive.reshape((-1, 1)), idx_negative.reshape((-1, 1)), src_match_coords, tgt_match_coords
#
#
# def extract_matching_feats(src, tgt, idx_positive, idx_negative):
#     # feature matrix
#     feats_matrix = src @ tgt.transpose(1, 0)
#
#     # matching features
#     feats_positive = feats_matrix[idx_positive]
#     feats_negative = feats_matrix[idx_negative]
#
#     return feats_positive, feats_negative
#
#
# class MetericLoss(nn.Module):
#
#     def __init__(self, m=0.25, gamma=80.):
#         super().__init__()
#
#         self.circle_weights = Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
#         self.circle_loss = CircleLoss(m, gamma)
#
#         self.overlap_weights = Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
#         self.overlap_loss = OverlapLoss()
#
#         self.match_weights = Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
#         self.match_loss = MatchLoss()
#
#         self.dist = nn.PairwiseDistance()
#         self.eps = 1e-7
#
#     def forward(self, src, tgt, src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords):
#         idx_positive, idx_negative, src_match_coords, tgt_match_coords = \
#             corresponding(src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords)
#         #
#         # # circle loss
#         # circle_loss = self.circle_loss(*extract_matching_feats(src, tgt, idx_positive, idx_negative))
#         # # overlap_loss = self.overlap_loss(src_match_coords, tgt_match_coords)
#         # # match_loss = self.match_loss(src_coords, tgt_coords)
#         #
#         # # loss = circle_loss * self.circle_weights + overlap_loss * self.overlap_weights + match_loss * self.match_weights
#         #
#         # loss = circle_loss * self.circle_weights
#         # return loss
#
#         return torch.sum(self.dist(src[idx_positive], tgt[id]) + self.eps, dim=0)
#

class MetericLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.circle_weights = Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
        self.circle_loss = CircleLoss()

        self.overlap_weights = Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
        self.overlap_loss = OverlapLoss()

        self.match_weights = Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
        self.match_loss = MatchLoss()

    def forward(self, src, tgt, src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords):
        circle_loss = self.circle_loss(src, tgt, src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords)
        #overlap_loss = self.overlap_loss(src, tgt, src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords)
        # match_loss = self.match_loss(src, tgt, src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords)

        loss = circle_loss * self.circle_weights
        return loss


if __name__ == "__main__":
    pass
