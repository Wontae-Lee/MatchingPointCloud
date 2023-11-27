import numpy as np
from kernel.kernel_point_generator import KernelPointGenerator
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
        self.implant = implant
        self.bone = bone
        self.number_of_layers = 0
        self.kernel_points = []
        self.smoothing_lengths = []
        self.sampled_implant = []
        self.center_implant_point_in_grid = [implant]
        self.sampled_bone = []
        self.center_bone_point_in_grid = [bone]

    def add_layer(self, layer: Layer):
        # count the number of layers
        self.number_of_layers += 1

        # add the kernel points to the list
        self.kernel_points.append(layer.kernel_point)

        # add the smoothing length to the list
        self.smoothing_lengths.append(layer.smoothing_length)

        # sampling the point clouds
        self.__sampling(layer.sampling_grid_size)

    def __sampling(self, sampling_grid_size: np.array):
        # Sample the point clouds
        implant_sampling = GridSampling(self.center_implant_point_in_grid[-1], sampling_grid_size)
        bone_sampling = GridSampling(self.center_bone_point_in_grid[-1], sampling_grid_size)

        # Add the sampled point clouds to the list
        self.sampled_implant.append(implant_sampling.sampled_point_cloud)
        self.sampled_bone.append(bone_sampling.sampled_point_cloud)

        # Add the center point of the sampling grid to the list
        self.center_implant_point_in_grid.append(implant_sampling.center_point_in_grid)
        self.center_bone_point_in_grid.append(bone_sampling.center_point_in_grid)

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
