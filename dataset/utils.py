import open3d as o3d
import numpy as np


def visualize_point_cloud(points: np.array, color: np.array = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd])

