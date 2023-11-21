import open3d as o3d
import numpy as np


def visualize_point_cloud(points: np.array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def where_is_min_value_location(array2d: np.array):
    # 최솟값의 위치 찾기
    min_value_location = np.where(array2d == np.min(array2d))
    min_value_location = np.hstack((min_value_location[0].reshape(-1, 1), min_value_location[1].reshape(-1, 1)))

    return min_value_location


def return_points_in_sample(sample, center):
    points = []
    indices = []
    for _, point_in_grid in enumerate(sample):
        x_grid = point_in_grid[0]
        y_grid = point_in_grid[1]
        z_grid = point_in_grid[2]

        for index, center_point in enumerate(center):
            x = center_point[0]
            y = center_point[1]
            z = center_point[2]

            if x_grid == x and y_grid == y and z_grid == z:
                indices.append(index)
                points.append(point_in_grid)
    return indices, points
