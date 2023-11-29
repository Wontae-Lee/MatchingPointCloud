import numpy as np
import copy
import open3d as o3d
from kernel.kernel_point_generator import KernelPointGenerator
from kernel.gaussian_kernel import GaussianKernel
from helpers.sampling import GridSampling

np.random.seed(0)


class Layer:

    def __init__(self, kernel_size: int, radius: float, smoothing_length: float):
        self.kernel_size = kernel_size
        self.radius = radius
        self.kernel_point = KernelPointGenerator(kernel_size, radius).kernel_points
        self.smoothing_length = smoothing_length
        self.sampling_grid_size = np.array([radius * 2, radius * 2, radius * 2])


class Model:

    def __init__(self, implant: np.array, bone: np.array):

        # implant and bone point clouds
        self.implant = implant
        self.bone = bone

        # number of layers
        self.number_of_layers = 0

        # sampled point clouds
        self.sampled_implant = []
        self.sampled_bone = []
        # center point of the sampling grid
        self.center_point_implant = []
        self.center_point_bone = []

        # feautures
        self.features_implant = []
        self.features_bone = []

    def add_layer(self, layer: Layer):
        # count the number of layers
        self.number_of_layers += 1

        if self.number_of_layers == 1:

            center_point_implant = self.implant
            center_point_bone = self.bone
        else:

            center_point_implant = self.center_point_implant[-1]
            center_point_bone = self.center_point_bone[-1]

        # Sample the point clouds
        implant_sampling = GridSampling(center_point_implant, layer.sampling_grid_size)
        bone_sampling = GridSampling(center_point_bone, layer.sampling_grid_size)

        # Samples
        sampled_implant = implant_sampling.sampled_point_cloud
        sampled_bone = bone_sampling.sampled_point_cloud

        # center points
        center_point_implant = implant_sampling.center_point_in_grid
        center_point_bone = bone_sampling.center_point_in_grid

        # features
        features_implant = GaussianKernel(kernel_point=layer.kernel_point,
                                          center_point_in_grid=center_point_implant,
                                          point_cloud_in_grid=sampled_implant,
                                          h=layer.smoothing_length).data
        features_bone = GaussianKernel(kernel_point=layer.kernel_point,
                                       center_point_in_grid=center_point_bone,
                                       point_cloud_in_grid=sampled_bone,
                                       h=layer.smoothing_length).data

        # logging
        # Add the sampled point clouds to the list
        self.sampled_implant.append(sampled_implant)
        self.sampled_bone.append(sampled_bone)

        # Add the center point of the sampling grid to the list
        self.center_point_implant.append(implant_sampling.center_point_in_grid)
        self.center_point_bone.append(bone_sampling.center_point_in_grid)

        # Add the features to the list
        self.features_implant.append(features_implant.sum(axis=2).sum(axis=1))
        self.features_bone.append(features_bone.sum(axis=2).sum(axis=1))

    @staticmethod
    def index_min_difference_features(features_implant, features_bone):

        # Get the difference
        difference = np.abs(features_implant.reshape(-1, 1) - features_bone.reshape(1, -1))

        # Get the index of the minimum difference
        min_index = np.unravel_index(np.argmin(difference, axis=None), difference.shape)

        return min_index

    @staticmethod
    def unpack(value_array, center_point):

        temp_value_array = copy.deepcopy(value_array)
        if type(temp_value_array) is float:
            temp_value_array = np.array([temp_value_array])

        # index list to store the corresponding points
        index_list = []

        for _, point in enumerate(temp_value_array):
            for index, __ in enumerate(center_point):

                if np.all(point == center_point[index]):
                    index_list.append(index)

        return index_list

    def fix(self, array):

        fix_array = np.zeros((0, 3))
        for index, point in enumerate(array):
            fix_array = np.vstack((fix_array, point))

        return fix_array

    def train(self):

        # Get the index of the last layer
        index = self.number_of_layers - 1
        min_index3 = self.index_min_difference_features(self.features_implant[index], self.features_bone[index])

        # sampling from deep layers
        # layer3
        sample_implant3 = self.sampled_implant[index][min_index3[0]]
        center_implant3 = self.center_point_implant[index - 1]

        sample_bone3 = self.sampled_bone[index][min_index3[1]]
        center_bone3 = self.center_point_bone[index - 1]

        index_implant3 = self.unpack(sample_implant3, center_implant3)
        index_bone3 = self.unpack(sample_bone3, center_bone3)

        features_implant3 = self.features_implant[index - 1][index_implant3]
        features_bone3 = self.features_bone[index - 1][index_bone3]

        # layer2
        min_index2 = self.index_min_difference_features(features_implant3, features_bone3)
        implant_index2 = self.unpack(features_implant3[min_index2[0]], self.features_implant[index - 1])
        bone_index2 = self.unpack(features_bone3[min_index2[1]], self.features_bone[index - 1])

        sample_implant2 = self.sampled_implant[index - 1][implant_index2]
        sample_implant2 = self.fix(sample_implant2)

        sample_bone2 = self.sampled_bone[index - 1][bone_index2]
        sample_bone2 = self.fix(sample_bone2)

        index_implant2 = self.unpack(sample_implant2, self.center_point_implant[index - 2])
        index_bone2 = self.unpack(sample_bone2, self.center_point_bone[index - 2])

        features_implant2 = self.features_implant[index - 2][index_implant2]
        features_bone2 = self.features_bone[index - 2][index_bone2]

        # layer1
        min_index1 = self.index_min_difference_features(features_implant2, features_bone2)
        implant_index1 = self.unpack(features_implant2[min_index1[0]], self.features_implant[index - 2])
        bone_index1 = self.unpack(features_bone2[min_index1[1]], self.features_bone[index - 2])

        sample_implant1 = self.sampled_implant[index - 2][implant_index1]
        sample_implant1 = self.fix(sample_implant1)

        sample_bone1 = self.sampled_bone[index - 2][bone_index1]
        sample_bone1 = self.fix(sample_bone1)

        # temp

        temp_sample_implant3 = self.sampled_implant[index][min_index3[0]]
        temp_sample_bone3 = self.sampled_bone[index][min_index3[1]]

        temp_index_implant3 = self.unpack(temp_sample_implant3, self.center_point_implant[index - 1])
        temp_index_bone3 = self.unpack(temp_sample_bone3, self.center_point_bone[index - 1])

        check_implant = self.sampled_implant[index - 1][temp_index_implant3]
        check_bone = self.sampled_bone[index - 1][temp_index_bone3]

        check_implant = self.fix(check_implant)
        check_bone = self.fix(check_bone)

        check2_implant = self.unpack(check_implant, self.center_point_implant[index - 2])
        check2_bone = self.unpack(check_bone, self.center_point_bone[index - 2])

        check3_implant = self.sampled_implant[index - 2][check2_implant]
        check3_bone = self.sampled_bone[index - 2][check2_bone]

        check3_implant = self.fix(check3_implant)
        check3_bone = self.fix(check3_bone)

        check = np.vstack((check3_implant, check3_bone))

        # visualize
        result = np.vstack((sample_implant1, sample_bone1))

        result_pcd = o3d.geometry.PointCloud()
        result_pcd.points = o3d.utility.Vector3dVector(result)
        result_pcd.paint_uniform_color([1, 0.706, 0])

        check_pcd = o3d.geometry.PointCloud()
        check_pcd.points = o3d.utility.Vector3dVector(check)
        check_pcd.paint_uniform_color([0, 0.651, 0.929])

        orginal = np.vstack((self.implant, self.bone))
        orginal_pcd = o3d.geometry.PointCloud()
        orginal_pcd.points = o3d.utility.Vector3dVector(orginal)
        orginal_pcd.paint_uniform_color([0, 0.124, 0.592])

        # 같은 feature가 너무 많다 커널 사이즈를 늘리니까 해결
        o3d.visualization.draw_geometries([check_pcd, orginal_pcd])
        #o3d.visualization.draw_geometries([result_pcd, orginal_pcd])


if __name__ == "__main__":
    implant = np.load('../dataset/geometry/implant.npy')
    bone = np.load('../dataset/geometry/bone.npy')

    model = Model(implant, bone)
    model.add_layer(Layer(128, 3, 0.5))
    model.add_layer(Layer(256, 6, 1.0))
    model.add_layer(Layer(512, 12, 2.0))
    model.train()
