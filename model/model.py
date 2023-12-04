import numpy as np
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

        # The raw point clouds correspond to each layer
        self.corresponding_raw_implant = []
        self.corresponding_raw_bone = []

    def add_layer(self, layer: Layer, saved: bool = False):

        # count the number of layers
        self.number_of_layers += 1

        # load data
        if saved:

            # load features
            features_implant = np.load(f'./features/features_implant{self.number_of_layers}.npy', allow_pickle=True)
            features_bone = np.load(f'./features/features_bone{self.number_of_layers}.npy', allow_pickle=True)

            # load sampled point clouds
            sampled_implant = np.load(f'./features/sampled_implant{self.number_of_layers}.npy', allow_pickle=True)
            sampled_bone = np.load(f'./features/sampled_bone{self.number_of_layers}.npy', allow_pickle=True)

            # load center points
            center_point_implant = np.load(f'./features/center_point_implant{self.number_of_layers}.npy',
                                           allow_pickle=True)
            center_point_bone = np.load(f'./features/center_point_bone{self.number_of_layers}.npy', allow_pickle=True)

        else:

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

            # save the features
            np.save(f'./features/features_implant{self.number_of_layers}.npy', features_implant)
            np.save(f'./features/features_bone{self.number_of_layers}.npy', features_bone)

            # save the sampled point clouds
            np.save(f'./features/sampled_implant{self.number_of_layers}.npy', sampled_implant)
            np.save(f'./features/sampled_bone{self.number_of_layers}.npy', sampled_bone)

            # save the center points
            np.save(f'./features/center_point_implant{self.number_of_layers}.npy', center_point_implant)
            np.save(f'./features/center_point_bone{self.number_of_layers}.npy', center_point_bone)

        # Add the sampled point clouds to the list
        self.sampled_implant.append(sampled_implant)
        self.sampled_bone.append(sampled_bone)

        # Add the center point of the sampling grid to the list
        self.center_point_implant.append(center_point_implant)
        self.center_point_bone.append(center_point_bone)

        # Add the features to the list
        self.features_implant.append(features_implant.sum(axis=2).sum(axis=1))
        self.features_bone.append(features_bone.sum(axis=2).sum(axis=1))

    @staticmethod
    def index_min_difference_features(features_implant, features_bone):

        # Get the difference
        difference = np.abs(features_implant.reshape(-1, 1) - features_bone.reshape(1, -1))

        # Get the index of the minimum difference
        min_indices = np.transpose(np.array(np.where(difference == np.min(difference))))

        return min_indices

    @staticmethod
    def corresponding(value_array, search_array):
        # index list to store the corresponding points
        index_list = []

        for _, value in enumerate(value_array):
            for index, __ in enumerate(search_array):

                if np.all(value == search_array[index]):
                    index_list.append(index)

        return np.array(index_list)

    @staticmethod
    def compact(array):
        """
        :param array: array to be compacted
        structure of the array
        sampled center points in a grid
        numpy arrays in list
        [np.array([1,2,3],
                  [4,5,6],
                  [7,8,9],
         np.array([10,11,12], ...)]
        """

        fix_array = np.zeros((0, 3))
        for index, point in enumerate(array):
            fix_array = np.vstack((fix_array, point))

        return fix_array

    def unpack(self, layer, index):

        # index[:, 0] : the minimum difference index of the implant
        # index[:, 1] : the minimum difference index of the bone
        sample_implant = self.sampled_implant[layer][index[:, 0]]
        sample_bone = self.sampled_bone[layer][index[:, 1]]

        # the sampled points are consist of the center points of the sampling grid
        # So, we need to find which center point is corresponding to the sampled points
        # The "center point" is the center point of the samples of the previous layer
        center_point_implant = self.center_point_implant[layer - 1]
        center_point_bone = self.center_point_bone[layer - 1]

        # compact the sampled points which have the minimum difference
        sample_implant = self.compact(sample_implant)
        sample_bone = self.compact(sample_bone)

        # find the corresponding the indices of the center points
        index_implant = self.corresponding(sample_implant, center_point_implant)
        index_bone = self.corresponding(sample_bone, center_point_bone)

        # unique
        index_implant = np.unique(index_implant)
        index_bone = np.unique(index_bone)
        return index_implant, index_bone

    def return_corresponding_features(self, layer, index_implant, index_bone):

        features_implant = self.features_implant[layer - 1][index_implant]
        features_bone = self.features_bone[layer - 1][index_bone]

        return features_implant, features_bone

    def train(self):

        # Get the index of the last layer
        layer = self.number_of_layers - 1

        # initialize
        index_min = None
        features_implant = self.features_implant[-1]
        features_bone = self.features_bone[-1]

        while True:

            if index_min is None:
                index_min = self.index_min_difference_features(features_implant, features_bone)
            else:
                index_min = self.index_min_difference_features(features_implant, features_bone)

            # Get the indice of the corresponding points from the sampled point clouds
            index_implant, index_bone = self.unpack(layer, index_min)

            # Get the corresponding features
            features_implant, features_bone = self.return_corresponding_features(layer, index_implant, index_bone)

            # Update the layer
            layer -= 1
            if layer == 0:

                index_min = self.index_min_difference_features(self.features_implant[0][index_implant],
                                                               self.features_bone[0][index_bone])


                raw_index_implant = self.compact(self.sampled_implant[0][index_implant][index_min[:, 0]])
                raw_index_bone = self.compact(self.sampled_bone[0][index_bone][index_min[:, 1]])

                # features가 같아도 다른 포인트 상위 레이어에서 다른 셈플에 들어갈 수 있으니까 그걸 제외해야한다.
                break

        # visualize
        implant = raw_index_implant
        bone = raw_index_bone

        return implant, bone


if __name__ == "__main__":
    implant = np.load('../dataset/geometry/implant.npy')
    bone = np.load('../dataset/geometry/bone.npy')

    model = Model(implant, bone)
    model.add_layer(Layer(256, 1, 0.5), saved=False)
    model.add_layer(Layer(512, 2, 1), saved=False)
    model.add_layer(Layer(1024, 4, 2.0), saved=False)
    imp, bon = model.train()

    implant[:, 2] -= 50
    imp[:, 2] -= 50

    # visualize
    print(imp.shape)
    print(bon.shape)

    all_points = np.vstack((imp, bon))
    all_points_pcd = o3d.geometry.PointCloud()
    all_points_pcd.points = o3d.utility.Vector3dVector(all_points)
    all_points_pcd.paint_uniform_color([0, 0.3, 0])
    # o3d.visualization.draw_geometries([all_points_pcd])
    raw_points = np.vstack((implant, bone))
    raw_points_pcd = o3d.geometry.PointCloud()
    raw_points_pcd.points = o3d.utility.Vector3dVector(raw_points)
    raw_points_pcd.paint_uniform_color([0, 0.991, 0.929])

    o3d.visualization.draw_geometries([all_points_pcd, raw_points_pcd])
