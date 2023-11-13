import numpy as np


class GridSampling:

    def __init__(self, point_cloud_path: str, grid_size: np.array, save_sample_path: str = None,
                 save_center_path: str = None):
        # Load the point cloud
        self.point_cloud = np.load(point_cloud_path)

        # define the grid size
        self.grid_size = grid_size

        # Define the grid indices to which each point belongs
        self.grid_indices = np.floor(self.point_cloud / grid_size).astype(np.int32)

        # Get the unique grid indices
        self.unique_grid_indices = np.unique(self.grid_indices, axis=0)

        # Initialize the sampled point cloud
        self.sampled_point_cloud = self.__sampling()

        # Save the sampled point cloud
        if save_sample_path is not None:
            self.save_sampled_point_cloud(save_sample_path)
        if save_center_path is not None:
            self.save_grid_indices_point_cloud(save_center_path)

    def __sampling(self):
        sampled_point_cloud = []

        for i in range(self.unique_grid_indices.shape[0]):
            # Create a mask for the current grid cell
            mask = np.zeros(self.point_cloud.shape[0], dtype=bool)
            mask[np.all(self.unique_grid_indices[i] == self.grid_indices, axis=1)] = True

            # Get the points in the current grid cell
            sampled_point_cloud.append(self.point_cloud[mask])

        # Convert the list to a numpy array
        sampled_point_cloud = np.array(sampled_point_cloud, dtype=object)

        return sampled_point_cloud

    def save_grid_indices_point_cloud(self, path: str):
        # Convert the grid indices to a point cloud
        grid_indices_point_cloud = self.unique_grid_indices * self.grid_size + self.grid_size / 2

        assert grid_indices_point_cloud.shape[-1] == 3, "The grid indices point cloud must be 3D"
        assert len(grid_indices_point_cloud.shape) == 2, "The shape of the grid indices point cloud must be (N, 3)"

        np.save(path, grid_indices_point_cloud)

    def save_sampled_point_cloud(self, path: str):
        # Save the sampled point cloud
        np.save(path, self.sampled_point_cloud, allow_pickle=True)


if __name__ == "__main__":
    pass
