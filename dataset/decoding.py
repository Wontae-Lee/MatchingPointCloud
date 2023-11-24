import numpy as np


class Decoding:

    def __init__(self,
                 implant: np.array,
                 implant_sample3: np.array,
                 implant_center2: np.array,
                 implant_sample2: np.array,
                 implant_center1: np.array,
                 implant_sample1: np.array,

                 implant_feature3: np.array,
                 implant_feature2: np.array,
                 implant_feature1: np.array,

                 bone: np.array,
                 bone_sample3: np.array,
                 bone_center2: np.array,
                 bone_sample2: np.array,
                 bone_center1: np.array,
                 bone_sample1: np.array,

                 bone_feature3: np.array,
                 bone_feature2: np.array,
                 bone_feature1: np.array,

                 ):
        """
        :param raw_point_cloud: 원본이 되는 point cloud
        :param implant_sample3: 가장 큰 부피로 sampling한 point cloud 이다 여기서 point cloud는 3차원이고, 그 상위 sample2에서의 center point 들이다.
        :param implant_center2: gridsampling한 sample2 grid들의 center point 들이다.
        :param implant_sample2: 두번째 로 큰 부피로 sampling한 point cloud 이다 여기서 point cloud는 3차원이고, 그 상위 sample1에서의 center point 들이다.
        :param implant_center1: gridsampling한 sample1 grid들의 center point 들이다.
        :param implant_sample1: raw point cloud를 gridsampling한 sample 이다
        :param implant_feature3: implant_feature3
        :param implant_feature2: implant_feature2
        :param implant_feature1: implant_feature1
        :param bone_feature3: bone_feature3
        :param bone_feature2: bone_feature2
        :param bone_feature1: bone_feature1
        """

        # 외부에서온 데이터
        self.implant = implant
        self.implant_sample3 = implant_sample3
        self.implant_center2 = implant_center2
        self.implant_sample2 = implant_sample2
        self.implant_center1 = implant_center1
        self.implant_sample1 = implant_sample1

        self.implant_feature3 = implant_feature3
        self.implant_feature2 = implant_feature2
        self.implant_feature1 = implant_feature1

        self.bone = bone
        self.bone_sample3 = bone_sample3
        self.bone_center2 = bone_center2
        self.bone_sample2 = bone_sample2
        self.bone_center1 = bone_center1
        self.bone_sample1 = bone_sample1

        self.bone_feature3 = bone_feature3
        self.bone_feature2 = bone_feature2
        self.bone_feature1 = bone_feature1

        # 내부에서 계산되는 데이터
        self.desampled_raw_point_cloud = np.zeros((0, 3))

        self.__analyis_feature()

    def __analyis_feature(self):
        # features: 각 샘필링된 grid에서 각각의 포인트 클라우드와 커널 포인트들 사이의 거리
        # features shape: (number of grid samples, number of kernel, number of points in a grid)
        # 각 grid 안에서 gaussian kernel을 적용한 후, 합이 feature가 된다.

        # features shape: (number of grid samples) 1D array로 만든다.
        implant_feature3 = self.implant_feature3.sum(axis=2).sum(axis=1).reshape(-1, 1)
        bone_feature3 = self.bone_feature3.sum(axis=2).sum(axis=1).reshape(1, -1)

        # features3 에서 가장 차이가 나지 않는 grid를 찾는다.
        result_feature3 = np.abs(implant_feature3 - bone_feature3)
        min_value_location3 = self.min_value_location(result_feature3)

        implant_indices2, _ = self.return_points_in_sample(self.implant_sample3[min_value_location3[0][0]],
                                                           self.implant_center2)

        bone_indices2, _ = self.return_points_in_sample(self.bone_sample3[min_value_location3[0][1]],
                                                        self.bone_center2)

        # features3 에서 찾은 grid에서 가장 차이가 적게 나는 grid를 찾아보자
        implant_feature2 = self.implant_feature2[implant_indices2].sum(axis=2).sum(axis=1).reshape(-1, 1)
        bone_feature2 = self.bone_feature2[bone_indices2].sum(axis=2).sum(axis=1).reshape(1, -1)

        result_feature2 = np.abs(implant_feature2 - bone_feature2)
        min_value_location2 = self.min_value_location(result_feature2)

        implant_indices1, _ = self.return_points_in_sample(self.implant_sample2[min_value_location2[0][0]],
                                                           self.implant_center1)

        bone_indices1, _ = self.return_points_in_sample(self.bone_sample2[min_value_location2[0][1]],
                                                        self.bone_center1)

        implant_feature1 = self.implant_feature1[implant_indices1].sum(axis=2).sum(axis=1).reshape(-1, 1)
        bone_feature1 = self.bone_feature1[bone_indices1].sum(axis=2).sum(axis=1).reshape(1, -1)

        result_feature1 = np.abs(implant_feature1 - bone_feature1)
        min_value_location1 = self.min_value_location(result_feature1)

        implant_indices0, _ = self.return_points_in_sample(self.implant_sample1[min_value_location1[0][0]],
                                                           self.implant)

        bone_indices0, _ = self.return_points_in_sample(self.bone_sample1[min_value_location1[0][1]],
                                                        self.bone)

        np.save('.implant_indices0.npy', implant_indices0)
        np.save('.bone_indices0.npy', bone_indices0)
    @staticmethod
    def min_value_location(array2d: np.array):
        # 최솟값의 위치 찾기
        min_value_location = np.where(array2d == np.min(array2d))
        min_value_location = np.hstack((min_value_location[0].reshape(-1, 1), min_value_location[1].reshape(-1, 1)))

        return min_value_location

    @staticmethod
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


if __name__ == '__main__':
    decoding = Decoding(
        implant=np.load('geometry/implant.npy', allow_pickle=True),
        implant_sample3=np.load('geometry/implant_sample3.npy', allow_pickle=True),
        implant_center2=np.load('geometry/implant_center2.npy', allow_pickle=True),
        implant_sample2=np.load('geometry/implant_sample2.npy', allow_pickle=True),
        implant_center1=np.load('geometry/implant_center1.npy', allow_pickle=True),
        implant_sample1=np.load('geometry/implant_sample1.npy', allow_pickle=True),

        implant_feature3=np.load('features/implant_feature3.npy', allow_pickle=True),
        implant_feature2=np.load('features/implant_feature2.npy', allow_pickle=True),
        implant_feature1=np.load('features/implant_feature1.npy', allow_pickle=True),

        bone=np.load('geometry/bone.npy', allow_pickle=True),
        bone_sample3=np.load('geometry/bone_sample3.npy', allow_pickle=True),
        bone_center2=np.load('geometry/bone_center2.npy', allow_pickle=True),
        bone_sample2=np.load('geometry/bone_sample2.npy', allow_pickle=True),
        bone_center1=np.load('geometry/bone_center1.npy', allow_pickle=True),
        bone_sample1=np.load('geometry/bone_sample1.npy', allow_pickle=True),

        bone_feature3=np.load('features/bone_feature3.npy', allow_pickle=True),
        bone_feature2=np.load('features/bone_feature2.npy', allow_pickle=True),
        bone_feature1=np.load('features/bone_feature1.npy', allow_pickle=True),
    )
