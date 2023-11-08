import torch
import numpy as np
import open3d as o3d

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


class NIMSqtt:
    def __init__(self, raw_path="./geometry/raw.stl"):
        self.raw = self.stl_to_point_cloud(raw_path)
        self.implant = self.raw[np.where(self.raw[:, 2] >= 0.045)]
        self.bone = self.raw[np.where(self.raw[:, 2] < 0.045)]

        self.save_point_cloud(self.raw, "geometry/raw.npy")
        self.save_point_cloud(self.implant, "geometry/implant.npy")
        self.save_point_cloud(self.bone, "geometry/bone.npy")

    @staticmethod
    def stl_to_point_cloud(stl_filename, numpoints=500000):
        # Load the STL file into a triangle mesh
        triangle_mesh = o3d.io.read_triangle_mesh(stl_filename)
        triangle_mesh.compute_vertex_normals()

        # Sample points from the triangle mesh
        point_cloud = triangle_mesh.sample_points_uniformly(number_of_points=numpoints)
        return np.asarray(point_cloud.points)

    @staticmethod
    def visualization(point_cloud):
        raw_pcd = o3d.geometry.PointCloud()
        raw_pcd.points = o3d.utility.Vector3dVector(point_cloud)

        o3d.visualization.draw_geometries([raw_pcd])

    @classmethod
    def save_point_cloud(cls, point_cloud, path):
        np.save(path, point_cloud)


if __name__ == "__main__":
    data = NIMSqtt()
