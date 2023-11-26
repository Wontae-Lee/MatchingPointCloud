import numpy as np
import torch


class GaussianKernel:

    def __init__(self,
                 kernel_point: np.array,
                 center_point_in_grid: np.array,
                 point_cloud_in_grid: np.array,
                 h: float,
                 save_path: str = None):
        self.kernel_point = kernel_point
        self.number_of_kernel = kernel_point.shape[0]
        self.center_point_in_grid = center_point_in_grid
        self.point_cloud_in_grid = point_cloud_in_grid
        self.h = h
        self.sigma = 1 / (torch.pi ** (3 / 2) * h ** 3)
        self.data = self.__compute()
        if save_path:
            np.save(save_path, self.data)

    def __compute(self):
        self.kernel_point = np.expand_dims(self.kernel_point, axis=0)
        self.center_point_in_grid = self.center_point_in_grid.reshape(-1, 1, 1, 3)
        self.kernel_point = self.center_point_in_grid + self.kernel_point
        self.kernel_point = self.kernel_point.reshape(self.center_point_in_grid.shape[0], -1, 3)

        data_chunk = []

        for index, grid_order in enumerate(self.point_cloud_in_grid):
            features = self.gaussian(grid_order, self.kernel_point[index])
            features = features.reshape(self.kernel_point.shape[1], -1)
            features = features.sum(axis=1)
            features = features.reshape(self.number_of_kernel, -1)
            data_chunk.append(features)

        data_chunk = np.array(data_chunk, dtype=object)

        return data_chunk

    def gaussian(self, point_cloud1: np.array, point_cloud2: np.array):
        pcd_torch1 = torch.from_numpy(point_cloud1).to('cuda').unsqueeze(0)
        pcd_torch2 = torch.from_numpy(point_cloud2).to('cuda').unsqueeze(0).permute(1, 0, 2)

        differences = pcd_torch1 - pcd_torch2
        result = self.sigma * torch.exp((self.h / torch.sqrt(torch.sum(differences ** 2, dim=2))) ** 2)
        result = result.cpu().numpy()

        return result
