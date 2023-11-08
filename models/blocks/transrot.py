import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class TransRot(nn.Module):

    def __init__(self):
        super(TransRot, self).__init__()

        '''
        Encoder blocks
        '''
        self.trans = Parameter(torch.zeros((1, 3), dtype=torch.float), requires_grad=True)
        self.rot_x = Parameter(torch.zeros((), dtype=torch.float), requires_grad=True)
        self.rot_y = Parameter(torch.zeros((), dtype=torch.float), requires_grad=True)
        self.rot_z = Parameter(torch.zeros((), dtype=torch.float), requires_grad=True)

    def forward(self, src, tgt, epoch):
        # transform
        print("")
        print("*****************************************************************")
        print("************************ Traning start **************************")
        print("*****************************************************************")
        print("")

        print("*****************************************************************")
        print("")
        print("trans\n", self.trans)
        print("")
        print("rot x axis\n", self.rot_x)
        print("")
        print("rot y axis\n", self.rot_y)
        print("")
        print("rot z axis\n", self.rot_z)

        # zero = torch.tensor(0.)
        # if epoch % 4 == 0:
        #     # rotation x-axis
        #     s_x = torch.sin(self.rot_x)
        #     c_x = torch.cos(self.rot_x)
        #     rot_x = torch.stack([torch.tensor([1., 0., 0.]),
        #                          torch.stack([zero, c_x, -s_x]),
        #                          torch.stack([zero, s_x, c_x])])
        #     src = src @ rot_x.t()
        # elif epoch % 4 == 1:
        #     # rotation y-axis
        #     s_y = torch.sin(self.rot_y)
        #     c_y = torch.cos(self.rot_y)
        #     rot_y = torch.stack([torch.stack([c_y, zero, s_y]),
        #                          torch.tensor([0., 1., 0.]),
        #                          torch.stack([zero, s_y, c_y])])
        #     src = src @ rot_y.t()
        # elif epoch % 4 == 2:
        #     # rotation z-axis
        #     s_z = torch.sin(self.rot_z)
        #     c_z = torch.cos(self.rot_z)
        #     rot_z = torch.stack([torch.stack([c_z, - s_z, zero]),
        #                          torch.stack([s_z, c_z, zero]),
        #                          torch.tensor([0., 0., 1.])])
        #     src = src @ rot_z.t()
        # else:
        #     src = src + self.trans
        zero = torch.tensor(0.)
        src = src + self.trans

        # rotation x-axis
        s_x = torch.sin(self.rot_x)
        c_x = torch.cos(self.rot_x)
        rot_x = torch.stack([torch.tensor([1., 0., 0.]),
                             torch.stack([zero, c_x, -s_x]),
                             torch.stack([zero, s_x, c_x])])
        src = src @ rot_x.t()

        # rotation y-axis
        s_y = torch.sin(self.rot_y)
        c_y = torch.cos(self.rot_y)
        rot_y = torch.stack([torch.stack([c_y, zero, s_y]),
                             torch.tensor([0., 1., 0.]),
                             torch.stack([zero, s_y, c_y])])
        src = src @ rot_y.t()

        # rotation z-axis
        s_z = torch.sin(self.rot_z)
        c_z = torch.cos(self.rot_z)
        rot_z = torch.stack([torch.stack([c_z, - s_z, zero]),
                             torch.stack([s_z, c_z, zero]),
                             torch.tensor([0., 0., 1.])])
        src = src @ rot_z.t()

        return src, tgt


if __name__ == '__main__':
    pass
