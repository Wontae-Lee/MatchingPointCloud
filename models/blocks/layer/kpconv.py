from torch.nn.parameter import Parameter
from dataset.kernel_point import KernelPoint
from models.blocks.utils import return_distance
import torch.nn as nn
import torch
import numpy as np
import os
from os.path import exists
from models.blocks.utils import neighbor_search


class KPConv(nn.Module):

    def __init__(self, in_channels, out_channels, kp_extention, radius, name,
                 kernel_size=15,
                 ndim=3,
                 stride=False,
                 ):
        super(KPConv, self).__init__()

        # Save parameters for forward
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kp_extent = kp_extention
        self.radius = radius
        self.name = name
        self.kernel_size = kernel_size
        self.ndim = ndim
        self.stride = stride

        # Initialize weights
        self.weights = Parameter(torch.rand((self.kernel_size, self.out_channels), dtype=torch.float32), requires_grad=True)
        self.kp = KernelPoint(in_channels, out_channels, kp_extention, radius, name).kp

        return

    def forward(self, src, tgt, src_coords, tgt_coords):
        src, src_coords = self.__forward("src", src, src_coords)
        tgt, tgt_coords = self.__forward("tgt", tgt, tgt_coords)

        return src, tgt, src_coords, tgt_coords

    def __forward(self, pcd_name, feats, coords):

        # sampling
        neighbor_bool_mat, center_pcd_ind, pcd_center_coords = self.__sampling(pcd_name, coords)

        # compute the kernel point influence
        kp_influence_list = self.__linear_kp_influence(pcd_name, pcd_center_coords, coords, neighbor_bool_mat)
        # compute the features
        all_feats = self._extract_feats(feats, neighbor_bool_mat, kp_influence_list)

        # out features
        out_feats = torch.matmul(all_feats, self.weights)
        out_feats = torch.sum(out_feats, dim=1)

        # normalization
        out_feats_max = torch.max(out_feats, dim=-1)
        out_feats = out_feats / out_feats_max[0].unsqueeze(1)

        return out_feats, pcd_center_coords

    def __sampling(self, pcd_name, coords):

        os.makedirs("./dataset/neighbor_bool_mat", exist_ok=True)
        os.makedirs("./dataset/center_pcd_ind", exist_ok=True)
        os.makedirs("./dataset/pcd_center_coords", exist_ok=True)

        # Create directory
        neighbor_bool_mat_file = f'./dataset/neighbor_bool_mat/{self.name}_{pcd_name}.pt'
        center_pcd_ind_file = f'./dataset/center_pcd_ind/{self.name}_{pcd_name}.pt'
        pcd_center_coords_file = f'./dataset/pcd_center_coords/{self.name}_{pcd_name}.pt'

        if not exists(neighbor_bool_mat_file):

            # boolen matrix [N, N]
            neighbor_bool_mat = neighbor_search(coords, coords, self.radius)

            # sample
            sample_list = []
            for pcd_number, ind in enumerate(neighbor_bool_mat):
                if torch.sum(ind) == 0:
                    sample_list.append(pcd_number)
                    continue

                hash_idx = torch.where(ind)[0].tolist()
                sample_list.append(hash_idx[0])

            # the index of the center point at the space
            center_pcd_ind = np.unique(np.array(sample_list))

            # the coordinates of the center point
            pcd_center_coords = coords[center_pcd_ind]

            # the nearby points around the center point
            neighbor_bool_mat = neighbor_bool_mat[center_pcd_ind]

            torch.save(neighbor_bool_mat, neighbor_bool_mat_file)
            torch.save(center_pcd_ind, center_pcd_ind_file)
            torch.save(pcd_center_coords, pcd_center_coords_file)

        else:
            neighbor_bool_mat = torch.load(neighbor_bool_mat_file)
            center_pcd_ind = torch.load(center_pcd_ind_file)
            pcd_center_coords = torch.load(pcd_center_coords_file)

        return neighbor_bool_mat, center_pcd_ind, pcd_center_coords

    def __linear_kp_influence(self, pcd_name, pcd_center_coords, coords, neighbor_bool_mat):

        os.makedirs("./dataset/kp_influence_list", exist_ok=True)

        kp_influence_list_file = f'./dataset/kp_influence_list/{self.name}_{pcd_name}.pt'
        if not exists(kp_influence_list_file):
            kp_influence_list = []
            for pcd_idx, neigbor_ind in enumerate(neighbor_bool_mat):
                # center point coordinates
                pcd_coords = pcd_center_coords[pcd_idx]

                # Move the center of the kernel point to the current center point.
                kp_moved_coords = self.kp.clone().detach().cuda()
                kp_moved_coords = kp_moved_coords.reshape((-1, 3))
                kp_moved_coords = kp_moved_coords + pcd_coords  # broadcasting

                # pcd neighbors
                pcd_neighbor_coords = coords[neigbor_ind]

                # 커널 포인트와 neighbor point 사이의 거리
                kp_influence = return_distance(kp_moved_coords.cpu(), pcd_neighbor_coords.cpu())
                kp_influence = torch.clamp((1 - kp_influence) / self.kp_extent, min=0.0)
                kp_influence_list.append(kp_influence)

            torch.save(kp_influence_list, kp_influence_list_file)
        else:
            kp_influence_list = torch.load(kp_influence_list_file)

        return kp_influence_list

    def _extract_feats(self, feats, neighbor_bool_mat, kp_influence_list):

        all_feats = torch.zeros((0, self.out_channels, self.kernel_size), dtype=torch.float32)
        for center_idx, neigbor_boolen_ind in enumerate(neighbor_bool_mat):
            feats_tmp = feats[neigbor_boolen_ind].unsqueeze(1) - feats[neigbor_boolen_ind].unsqueeze(2)
            feats_tmp = torch.clamp(feats_tmp, min=0.000001)
            feats_tmp = torch.sqrt(feats_tmp ** 2)

            feats_tmp = torch.sum(feats_tmp, dim=-1)
            feats_tmp = torch.sum(feats_tmp, dim=-1)

            feats_tmp = feats_tmp * kp_influence_list[center_idx].float().cuda()
            feats_tmp = torch.sum(feats_tmp, dim=1)
            feats_tmp = feats_tmp.reshape((1, self.out_channels, self.kernel_size))

            # 각 center idx마다 feature 쌓기
            all_feats = torch.vstack((all_feats, feats_tmp))
            all_feats = all_feats.type(torch.float32)

        return all_feats


if __name__ == "__main__":
    pass
