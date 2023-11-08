import numpy as np
import open3d as o3d


class GridSampling:

    def __init__(self, pcd_path: str, grid_size: np.array):
        # Load the point cloud
        self.pcd = np.load(pcd_path)

        # define the grid size
        self.grid_size = grid_size

        # Define the grid indices to which each point belongs
        self.grid_indices = (self.pcd // grid_size).astype(int)

        # Get the unique grid indices
        self.unique_grid_indices = np.unique(self.grid_indices, axis=0)

        # Initialize the sampled point cloud
        self.sampled_pcd = []

        # Perform the sampling
        self.__sampling()

    def __sampling(self):
        for i in range(self.unique_grid_indices.shape[0]):
            # Create a mask for the current grid cell
            mask = np.zeros(self.pcd.shape[0], dtype=bool)
            mask[np.where(np.all(self.unique_grid_indices[i] == self.grid_indices, axis=1))] = True

            # Get the points in the current grid cell
            self.sampled_pcd.append(self.pcd[mask])

        # Convert the list to a numpy array
        self.sampled_pcd = np.array(self.sampled_pcd, dtype=object)

    def grid_indices_to_point_cloud(self):
        # Convert the grid indices to a point cloud
        return self.unique_grid_indices * self.grid_size + self.grid_size / 2

    def save_sampled_pcd(self, path: str):
        # Save the sampled point cloud
        np.save(path, self.sampled_pcd)

    def visualize_grid_sampled_pcd(self):
        # Visualize the sampled point cloud
        pcd = o3d.geometry.PointCloud()

        # Convert the grid indices to a point cloud
        grid_sampled_pcd = self.grid_indices_to_point_cloud()

        # Set the point cloud
        pcd.points = o3d.utility.Vector3dVector(grid_sampled_pcd)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    # Define the grid size
    test_grid_size = np.array([0.005, 0.005, 0.005])

    # Define the path to the point cloud
    test_pcd_path = "geometry/raw.npy"

    # Create the grid sampling object
    grid_sampling = GridSampling(test_pcd_path, test_grid_size)

    # Save the sampled point cloud
    grid_sampling.save_sampled_pcd("../geometry/grid_sampled.npy")

    # Visualize the sampled point cloud
    grid_sampling.visualize_grid_sampled_pcd()
