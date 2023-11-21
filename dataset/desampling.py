import numpy as np
from .utils import return_points_in_sample


class DeSampling:

    def __init__(self,
                 raw_point_cloud: np.array,
                 sample3: np.array,
                 center2: np.array,
                 sample2: np.array,
                 center1: np.array,
                 sample1: np.array,
                 ):
        """
        :param raw_point_cloud: 원본이 되는 point cloud
        :param sample3: 가장 큰 부피로 sampling한 point cloud 이다 여기서 point cloud는 3차원이고, 그 상위 sample2에서의 center point 들이다.
        :param center2: gridsampling한 sample2 grid들의 center point 들이다.
        :param sample2: 두번째 로 큰 부피로 sampling한 point cloud 이다 여기서 point cloud는 3차원이고, 그 상위 sample1에서의 center point 들이다.
        :param center1: gridsampling한 sample1 grid들의 center point 들이다.
        :param sample1: raw point cloud를 gridsampling한 sample 이다
        """
        self.raw_point_cloud = raw_point_cloud
        self.sample3 = sample3
        self.center2 = center2
        self.sample2 = sample2
        self.center1 = center1
        self.sample1 = sample1

        self.desampled_point_cloud = np.zeros((0, 3))

    def get_desampled_point_cloud(self, subsample_grid_number):

        indices, points = return_points_in_sample(self.sample3[subsample_grid_number], self.center2)

        for index in indices:

            indices_in_grid, points_in_grid = return_points_in_sample(self.sample2[index], self.center1)

            for index_raw_point_cloud in indices_in_grid:

                for point in self.sample1[index_raw_point_cloud]:
                    self.desampled_point_cloud = np.vstack((self.desampled_point_cloud, point))

        return self.desampled_point_cloud


if __name__ == '__main__':
    pass
