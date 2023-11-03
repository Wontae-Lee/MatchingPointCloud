from torch.nn.parameter import Parameter
from os.path import exists
from models.blocks.utils import create_3d_rotations
import torch
import os

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class KernelPoint:

    def __init__(self, in_channels, out_channels, kp_extention, radius, name,
                 kernel_size=15,
                 ndim=3,
                 ):

        # Save parameters for forward
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kp_extent = kp_extention
        self.radius = radius
        self.kernel_size = kernel_size
        self.ndim = ndim
        self.name = name

        # the total number of kernel points
        self.total_nkp = kernel_size * out_channels

        # Initialize kernel points
        self.kp = self._init()

        return

    def _init(self):
        os.makedirs("./dataset/kernel_points", exist_ok=True)

        # Create directory
        kp_file = f'./dataset/kernel_points/{self.name}.pt'

        if not exists(kp_file):
            # Create kernel points
            kp = self._generate_norm()

            # update kernel points according to gradient
            kp = self._update(kp)

            # Rotate the kernel points randomly
            kp = self._random_rotate(kp)

            torch.save(kp, kp_file)
        else:
            kp = torch.load(kp_file)

        return kp

    def _generate_norm(self):

        # value range: -1 < random values < 1
        kp = torch.rand(self.total_nkp, self.ndim) * 2 - 1

        while True:

            # new kernel points that is going to be add to the current kernel points
            new_kp = torch.rand(self.total_nkp, self.ndim) * 2 - 1
            kp = torch.vstack((kp, new_kp))

            # distance between each kernel point(x,y,z) and Origin(0,0,0)
            dist = torch.sum(kp ** 2, dim=1)

            # Make the diameter below 1
            kp = kp[dist < 0.5, :]

            # current the number of points
            current_nkp = kp.size(0)

            if current_nkp > self.total_nkp:
                # If the number of current kernel points is bigger than the total number of kernel points
                # then break the while roof. Now distances between each kernel point and origin are shorter than 0.5.
                # which mean all the diameters are shorter than 1.0.
                break

        # Extract only the desired number of kernel points
        kp = kp[:self.total_nkp]
        kp = kp.reshape((self.kernel_size, self.out_channels, self.ndim))

        # The kernel point at the origin must be involved
        kp[:, 0, :] *= 0

        return kp

    def _update(self, kp):
        """
        The position of the points is random,
        but they should be evenly distributed.  It's a function to do that.
        :param kp: kernel points
        :return: kernel points
        """

        # gradient to compare new gradient
        kp_grad_norm = torch.zeros((self.kernel_size, self.out_channels))

        # Gradient threshold to stop optimization
        thresh = 1e-5

        # moving options
        self.moving_factor = 0.01
        self.continuous_moving_decay = 0.9995
        self.clip = Parameter(torch.tensor(0.05).float(), requires_grad=False)
        for ITER in range(1000):
            # Derivative of the sum of potentials of all points
            # A.shape: (kernel_size, number of kernel, 1, number of dimensions)
            A = kp.unsqueeze(2)

            # B.shape: (kernel_size, 1, number of kernel,  number of dimensions)
            B = kp.unsqueeze(1)

            # The sum of squared distance
            interd2 = torch.sum((A - B) ** 2, dim=-1)

            # Compute gradient
            inter_grads = (A - B) / (interd2.unsqueeze(-1) ** (3 / 2) + 1e-6)
            inter_grads = torch.sum(inter_grads, dim=1)

            # Derivative of the radius potential
            circle_grads = 10 * kp

            # All gradients
            grad = inter_grads + circle_grads

            # Compute norm of gradients
            kp_current_grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=-1))

            # terminate sequence
            if torch.max(torch.abs(kp_grad_norm[:, 1:] - kp_current_grad_norm[:, 1:])) < thresh:
                break
            elif torch.max(torch.abs(kp_grad_norm - kp_current_grad_norm)) < thresh:
                break

            kp_grad_norm = kp_current_grad_norm

            # updatate kernel points according to gradient
            kp = self._move(kp, grad, kp_current_grad_norm)

        return kp

    def _move(self, kp, grad, grad_norm):
        """
        Function to move the kernel points according to the gradients
        :param kp: kernel points
        :param grad: gradient
        :param grad_norm: normalized gradient
        :return: kernel points
        """

        # get moving distance
        moving_dists = torch.minimum(self.moving_factor * grad_norm, self.clip)
        moving_dists[:, 0] = 0

        kp -= moving_dists.unsqueeze(-1) * grad / (grad_norm + 1e-6).unsqueeze(-1)
        self.moving_factor *= self.continuous_moving_decay

        return kp

    def _random_rotate(self, kp):
        # Random roations for the kernel
        theta = torch.rand(1) * 2 * torch.pi
        phi = (torch.rand(1) - 0.5) * torch.pi

        # Create the first vector in carthesian coordinates
        u = torch.tensor([torch.cos(theta) * torch.cos(phi), torch.sin(theta) * torch.cos(phi), torch.sin(phi)])

        # Choose a random rotation angle
        alpha = torch.rand(1) * 2 * torch.pi

        # Create the rotation matrix with this vector and angle
        R = create_3d_rotations(torch.reshape(u, (1, -1)), torch.reshape(alpha, (1, -1)))[0]
        R = R.float()

        # Scale kernels
        kp = self.radius * kp

        # Rotate kernels
        kp = torch.matmul(kp, R)

        return kp.float()


if __name__ == "__main__":
    pass
