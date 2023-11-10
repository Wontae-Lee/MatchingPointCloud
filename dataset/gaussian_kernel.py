import torch
import numpy as np


class Kernel:

    def __init__(self, kernel_points: np.array, geometry_points: np.array, h: float, save_path: str = None):
        self.kernel_points = kernel_points
        self.geometry_points = geometry_points
        self.h = h
        self.sigma = 1 / (torch.pi ** (3 / 2) * h ** 3)
        self.data = self.__compute()
        if save_path:
            np.save(save_path, self.data)

    def __compute(self):

        data_chunk = []
        for pcd1 in self.kernel_points:
            for pcd2 in self.geometry_points:
                data_chunk.append([self.gaussian(pcd1, pcd2)])

        data_chunk = np.array(data_chunk, dtype=object)

        return data_chunk

    def gaussian(self, point_cloud1: np.array, point_cloud2: np.array):
        pcd_torch1 = torch.from_numpy(point_cloud1).to('cuda').unsqueeze(0)
        pcd_torch2 = torch.from_numpy(point_cloud2).to('cuda').unsqueeze(0).permute(1, 0, 2)

        differences = pcd_torch1 - pcd_torch2
        result = self.sigma * torch.exp((self.h / torch.sqrt(torch.sum(differences ** 2, dim=2))) ** 2)
        result = result.to_numpy()

        return result


if __name__ == "__main__":
    pass
