import torch.nn as nn
import torch
from models.blocks.utils import return_distance


def extract_sp_sn(feats, prob_i, prob_j, coords, gnn_coords):
    prob_i = torch.sum(prob_i.permute(1, 2, 0), dim=2)
    prob_j = torch.sum(prob_j.permute(2, 1, 0), dim=2)

    idx_i = prob_i.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)
    idx_j = prob_j.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)

    match_coords = gnn_coords[idx_i == idx_j]
    unmatch_coords = gnn_coords[idx_i != idx_j]

    dist_positive = return_distance(match_coords, coords)
    dist_negative = return_distance(unmatch_coords, coords)

    idx_positive = dist_positive.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)
    idx_negative = dist_negative.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)

    return feats[idx_positive], feats[idx_negative]


class CircleLoss(nn.Module):
    def __init__(self):
        super(CircleLoss, self).__init__()

        self.delta_p = 0.1
        self.delta_n = 1.4

    def forward(self, src, tgt, src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords):
        loss_p = self.__forward(src, tgt, src_prob, tgt_prob, src_coords, tgt_coords, src_gnn_coords, tgt_gnn_coords)
        loss_q = self.__forward(tgt, src, tgt_prob, src_prob, tgt_coords, src_coords, tgt_gnn_coords, tgt_gnn_coords)

        loss = (loss_p + loss_q) / 2

        return loss

    def __forward(self, feats_i, feats_j, prob_i, prob_j, coords_i, coords_j, gnn_coords_i, gnn_coords_j):
        # the number of feature point
        N = prob_i.size(0)

        # sp_i, sn_i = extract_sp_sn(feats_i, prob_i, prob_j, coords_i, gnn_coords_i)
        # sp_j, sn_j = extract_sp_sn(feats_j, prob_j, prob_i, coords_j, gnn_coords_j)
        #
        # dist = nn.PairwiseDistance()
        # loss_i = torch.logsumexp(dist(sp_i, sp_j) - self.delta_p, dim=0)
        # loss_j = torch.logsumexp(self.delta_n - dist(sn_i, sn_j), dim=0)

        return loss_i + loss_j


if __name__ == "__main__":
    pass
