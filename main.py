from models.architecture.kpfcnn import KPFCNN
from models.loss.metric_loss import MetericLoss
from torch import optim
import torch
from models.blocks.utils import trans_rot_ptx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class Model(object):

    def __init__(self,
                 model=KPFCNN(),
                 loss=MetericLoss(),
                 learning_rate=0.001,
                 momentum=0.2,
                 weight_decay=0.9995):
        self.model = model
        self.loss = loss

        # Set optimizer
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=learning_rate,
                                   momentum=momentum,
                                   weight_decay=weight_decay)

        # dataset
        # features are equal to coordinates at the beginning
        self.src = torch.load("./dataset/source/src.pt").cuda().float()
        self.tgt = torch.load("./dataset/target/tgt.pt").cuda().float()

        # deepcopy to prevent to be changed.
        self.src_coords = self.src.clone().detach()
        self.tgt_coords = self.tgt.clone().detach()

    def train(self, num_of_epochs=10):
        print('start training...')

        for epoch in range(num_of_epochs):
            # find the trans and rotation weights
            self.optimizer.zero_grad()
            # forward to get features
            # prob: correspoinding probablity that point match to the other point in other point clouds
            # [n_head, N, M]
            src, tgt, src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords = self.model(src=self.src.clone().detach(),
                                                                                                              tgt=self.tgt.clone().detach(),
                                                                                                              src_coords=self.src_coords.clone().detach(),
                                                                                                              tgt_coords=self.tgt_coords.clone().detach(), epoch=epoch)

            # Loss function
            loss = self.loss(src, tgt, src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords)

            loss.backward()
            self.optimizer.step()

            trans = self.model.trans_rot.trans.clone().detach()
            rot_x = self.model.trans_rot.rot_x.clone().detach()
            rot_y = self.model.trans_rot.rot_y.clone().detach()
            rot_z = self.model.trans_rot.rot_z.clone().detach()

            self.src = trans_rot_ptx(self.src_coords.clone().detach(), trans, rot_x, rot_y, rot_z)
            self.src_coords = self.src.clone().detach()

        # finish all epoch
        print("Training finish!")


def main():
    model = Model()
    model.train(10)


if __name__ == '__main__':
    main()
