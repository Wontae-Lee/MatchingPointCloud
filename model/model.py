import numpy as np
from kernel.kernel_point_generator import KernelPointGenerator
from helpers.sampling import GridSampling


class Layer:

    def __init__(self, kernel_size: int, radius: float):
        self.kernel_size = kernel_size
        self.radius = radius
        self.kernel_point = KernelPointGenerator(kernel_size, radius).kernel_points
        self.sampling_grid_size = np.array([radius * 2, radius * 2, radius * 2])


class Model:

    def __init__(self, implant: np.array, bone: np.array):
        self.implant = implant
        self.bone = bone
        self.number_of_layers = 0
        self.kernel_points = []
        self.sampled_implant = [implant]
        self.sampled_bone = [bone]

    def add_layer(self, layer: Layer):
        # count the number of layers
        self.number_of_layers += 1

        # add the kernel points to the list
        self.kernel_points.append(layer.kernel_point)

        # sampling the point clouds
        self.__sampling(layer.sampling_grid_size)

    def __sampling(self, sampling_grid_size: np.array):
        # Sample the point clouds
        implant_sampling = GridSampling(self.sampled_implant[-1], sampling_grid_size).sampled_point_cloud
        bone_sampling = GridSampling(self.sampled_bone[-1], sampling_grid_size).sampled_point_cloud

        # Add the sampled point clouds to the list
        self.sampled_implant.append(implant_sampling)
        self.sampled_bone.append(bone_sampling)

    def train(self):
        pass


if __name__ == "__main__":
    implant = np.load('../dataset/geometry/implant.npy')
    bone = np.load('../dataset/geometry/bone.npy')

    model = Model(implant, bone)
