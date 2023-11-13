import numpy as np
import open3d as o3d


class Data:
    def __init__(self,
                 raw_path="./geometry/raw.stl",
                 number_of_points=500000,
                 splicing_height=0.05,
                 unit="mm",
                 save_raw=None,
                 save_implant=None,
                 save_bone=None):

        self.raw = self.stl_to_point_cloud(raw_path, number_of_points)
        self.implant = self.raw[np.where(self.raw[:, 2] >= splicing_height)]
        self.bone = self.raw[np.where(self.raw[:, 2] < splicing_height)]
        if unit == "mm":
            self.raw *= 1000
            self.implant *= 1000
            self.bone *= 1000

        if save_raw is not None:
            self.save_point_cloud(self.raw, save_raw)
        if save_implant is not None:
            self.save_point_cloud(self.implant, save_implant)
        if save_bone is not None:
            self.save_point_cloud(self.bone, save_bone)

    @staticmethod
    def stl_to_point_cloud(raw_path, numpoints=500000):
        # Load the STL file into a triangle mesh
        triangle_mesh = o3d.io.read_triangle_mesh(raw_path)
        triangle_mesh.compute_vertex_normals()

        # Sample points from the triangle mesh
        point_cloud = triangle_mesh.sample_points_uniformly(number_of_points=numpoints)
        return np.asarray(point_cloud.points)

    @classmethod
    def save_point_cloud(cls, point_cloud, path):
        np.save(path, point_cloud)


if __name__ == "__main__":
    pass
