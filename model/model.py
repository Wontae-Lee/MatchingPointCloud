import numpy as np
from kernel.kernel_point_generator import KernelPointGenerator
from kernel.gaussian_kernel import GaussianKernel
from helpers.sampling import GridSampling


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

    def train(self):

        # Get the index of the last layer
        index = self.number_of_layers

        while True:
            # current index
            index -= 1

            # Check if the number of layers is zero
            if index == 0:
                break


if __name__ == "__main__":
    implant = np.load('../dataset/geometry/implant.npy')
    bone = np.load('../dataset/geometry/bone.npy')

    model = Model(implant, bone)
    model.add_layer(Layer(64, 4, 0.25))
    model.add_layer(Layer(128, 8, 0.50))
    model.train()
