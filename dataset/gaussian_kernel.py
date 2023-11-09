import torch
import numpy as np


class Kernel:

    def __init__(self, kernel_points: np.array, geometry_points: np.array, h: float):
        self.kernel_points = kernel_points
        self.geometry_points = geometry_points
        self.h = h
        self.sigma = 1/ (torch.pi**(3/2) * h**3)

    def compute(self):
        test_result = self.gaussian(self.kernel_points[0], self.geometry_points[0])
        print(test_result)

    def gaussian(self, point_cloud1: np.array, point_cloud2: np.array):
        pcd_torch1 = torch.from_numpy(point_cloud1).to('cuda').unsqueeze(0)
        pcd_torch2 = torch.from_numpy(point_cloud2).to('cuda').unsqueeze(0).permute(1, 0, 2)

        differences = pcd_torch1 - pcd_torch2
        distances = torch.sqrt(torch.sum(differences ** 2, dim=2))

        q_term = self.h / distances
        e_term = torch.exp(-q_term ** 2)
        return self.sigma * e_term


if __name__ == "__main__":
    sampled_points = np.load('../dataset/geometry/grid_sampled.npy', allow_pickle=True)
    kernel_points = np.load('../dataset/kernel_points/kernel_points.npy', allow_pickle=True)

    kernel = Kernel(kernel_points, sampled_points, 0.1)
    check = kernel.gaussian(kernel_points[0], sampled_points[0])
    print(check)