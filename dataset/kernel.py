import torch
import numpy as np
import open3d as o3d


class Kernel:

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


class KernelPoint:

    def __init__(self, kernel_size, radius):
        # define the kernel size and the radius of the kernel
        self.kernel_size = kernel_size
        self.radius = radius

        # initialize the kernel points
        self.kernel_points = self._init()

    def _init(self):
        # Create a kernel consist of points
        kernel = self.__create_icosahedron_kernel()

        # duplicate the kernel to make kernels
        kernel_points = np.tile(kernel, (self.kernel_size, 1, 1))

        # make the distance between the center point and the vertices equal to radius
        kernel_points *= self.radius

        # Apply the random rotation to each kernel
        for i in range(self.kernel_size):
            kernel_points[i] = self.random_rotation_matrix().dot(kernel_points[i].T).T

        return kernel_points

    @staticmethod
    def random_rotation_matrix():
        random_angles = np.random.uniform(0, 2 * np.pi, 3)  # Random angles in radians
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(random_angles[0]), -np.sin(random_angles[0])],
                        [0, np.sin(random_angles[0]), np.cos(random_angles[0])]])
        R_y = np.array([[np.cos(random_angles[1]), 0, np.sin(random_angles[1])],
                        [0, 1, 0],
                        [-np.sin(random_angles[1]), 0, np.cos(random_angles[1])]])
        R_z = np.array([[np.cos(random_angles[2]), -np.sin(random_angles[2]), 0],
                        [np.sin(random_angles[2]), np.cos(random_angles[2]), 0],
                        [0, 0, 1]])
        rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
        return rotation_matrix

    @staticmethod
    def __create_icosahedron_kernel():
        # create icosahedron
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 0, 0],
            [0, 1, phi],
            [0, 1, -phi],
            [0, -1, phi],
            [0, -1, -phi],
            [1, phi, 0],
            [1, -phi, 0],
            [-1, phi, 0],
            [-1, -phi, 0],
            [phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, 1],
            [-phi, 0, -1]
        ])

        # make the distance between the center point and the vertices equal to 1
        vertices /= np.sqrt(np.sum(vertices[1] ** 2))

        return vertices

    def save_kernel_point(self, path: str):
        np.save(path, self.kernel_points)

    def visualization(self):
        pcd = o3d.geometry.PointCloud()
        all_kernel_points = self.kernel_points.reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(all_kernel_points)
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    pass
