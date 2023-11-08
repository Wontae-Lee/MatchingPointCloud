import torch
import torch.nn as nn
import torch.nn.functional as tnf
from models.blocks.simple import Simple
from models.blocks.resnet_bottleneck import ResnetB, ResnetA
from models.blocks.layer.conv1d import Conv1d
from models.blocks.gnn import GNN
from models.blocks.attention import CrossAttention
from models.blocks.nearest_up import NearestUp
from models.blocks.transrot import TransRot

torch.autograd.set_detect_anomaly(True)


class KPFCNN(nn.Module):

    def __init__(self, radius=0.03, num_of_in_channels=64, num_of_gnn_channels=256):
        super(KPFCNN, self).__init__()

        '''
        parameters
        '''

        self.encoder_shortcut = []
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        self.trans_rot = TransRot()

        '''
        Encoder blocks
        '''

        self.simple = Simple(in_channels=1, out_channels=num_of_in_channels, radius=radius, name="simple")
        self.resnetb1 = ResnetB(in_channels=num_of_in_channels, out_channels=num_of_in_channels * 2, radius=radius, name="resnetb1")
        self.resneta1 = ResnetA(in_channels=num_of_in_channels * 2, out_channels=num_of_in_channels * 2, radius=radius * 2, name="resneta1")

        self.resnetb2 = ResnetB(in_channels=num_of_in_channels * 2, out_channels=num_of_in_channels * 2, radius=radius * 2, name="resnetb2")
        self.resnetb3 = ResnetB(in_channels=num_of_in_channels * 2, out_channels=num_of_in_channels * 2 * 2, radius=radius * 2, name="resnetb3")
        self.resneta2 = ResnetA(in_channels=num_of_in_channels * 2 * 2, out_channels=num_of_in_channels * 2 * 2, radius=radius * 2 * 2, name="resneta2")

        self.resnetb4 = ResnetB(in_channels=num_of_in_channels * 2 * 2, out_channels=num_of_in_channels * 2 * 2, radius=radius * 2 * 2, name="resnetb4")
        self.resnetb5 = ResnetB(in_channels=num_of_in_channels * 2 * 2, out_channels=num_of_in_channels * 2 * 2 * 2, radius=radius * 2 * 2, name="resnetb5")
        self.resneta3 = ResnetA(in_channels=num_of_in_channels * 2 * 2 * 2, out_channels=num_of_in_channels * 2 * 2 * 2, radius=radius * 2 * 2 * 2, name="resneta3")

        self.resnetb6 = ResnetB(in_channels=num_of_in_channels * 2 * 2 * 2, out_channels=num_of_in_channels * 2 * 2 * 2, radius=radius * 2 * 2 * 2, name="resnetb6")
        self.resnetb7 = ResnetB(in_channels=num_of_in_channels * 2 * 2 * 2, out_channels=num_of_in_channels * 2 * 2 * 2 * 2, radius=radius * 2 * 2 * 2, name="resnetb7")

        '''
        The deepest layer
        '''

        self.conv1d_intro = Conv1d(in_channels=num_of_in_channels * 2 * 2 * 2 * 2, out_channels=num_of_gnn_channels, kernel_size=1)
        self.gnn_intro = GNN(gnn_features_dimension=num_of_gnn_channels, dgcnn_kernel_size=2)
        self.cross_attention = CrossAttention(gnn_features_dimension=num_of_gnn_channels, num_of_heads=4)
        self.gnn_outro = GNN(gnn_features_dimension=num_of_gnn_channels, dgcnn_kernel_size=2)
        self.conv1d_outro = Conv1d(in_channels=num_of_gnn_channels, out_channels=num_of_gnn_channels, kernel_size=1)

        '''
        Decoder blocks
        '''
        self.nearest_up1 = NearestUp()
        self.conv1d_decoder1 = Conv1d(in_channels=num_of_gnn_channels + 2, out_channels=num_of_gnn_channels // 2 + 1, kernel_size=1)
        self.nearest_up2 = NearestUp()
        self.conv1d_decoder2 = Conv1d(in_channels=num_of_gnn_channels // 2 + 1, out_channels=num_of_gnn_channels // 4, kernel_size=1)
        self.nearest_up3 = NearestUp()
        self.conv1d_decoder3 = Conv1d(in_channels=num_of_gnn_channels // 4, out_channels=num_of_gnn_channels // 8 + 1, kernel_size=1)

    def forward(self, src, tgt, src_coords, tgt_coords, epoch):
        src, tgt = self.trans_rot(src, tgt, epoch)

        src, tgt, src_coords, tgt_coords = self.simple(src, tgt, src_coords, tgt_coords)

        src, tgt, src_coords, tgt_coords = self.resnetb1(src, tgt, src_coords, tgt_coords)
        src, tgt, src_coords, tgt_coords = self.resneta1(src, tgt, src_coords, tgt_coords)
        src, tgt = self.__shortcut_features(src, tgt, src_coords, tgt_coords)

        src, tgt, src_coords, tgt_coords = self.resnetb2(src, tgt, src_coords, tgt_coords)
        src, tgt, src_coords, tgt_coords = self.resnetb3(src, tgt, src_coords, tgt_coords)
        src, tgt, src_coords, tgt_coords = self.resneta2(src, tgt, src_coords, tgt_coords)
        src, tgt = self.__shortcut_features(src, tgt, src_coords, tgt_coords)

        src, tgt, src_coords, tgt_coords = self.resnetb4(src, tgt, src_coords, tgt_coords)
        src, tgt, src_coords, tgt_coords = self.resnetb5(src, tgt, src_coords, tgt_coords)
        src, tgt, src_coords, tgt_coords = self.resneta3(src, tgt, src_coords, tgt_coords)

        src, tgt = self.__shortcut_features(src, tgt, src_coords, tgt_coords)

        src, tgt, src_coords, tgt_coords = self.resnetb6(src, tgt, src_coords, tgt_coords)
        src, tgt, src_coords, tgt_coords = self.resnetb7(src, tgt, src_coords, tgt_coords)

        # The deepest blocks
        src, tgt = self.conv1d_intro(src, tgt)
        src, tgt = self.gnn_intro(src, tgt, src_coords, tgt_coords)
        src, tgt, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords = self.cross_attention(src, tgt, src_coords, tgt_coords)
        src, tgt = self.gnn_outro(src, tgt, src_coords, tgt_coords)
        src, tgt = self.conv1d_outro(src, tgt)

        src_scores, tgt_scores = self.__scores(src, tgt)
        src_saliency, tgt_saliency = self.__scores_saliency(src, tgt, src_scores, tgt_scores)
        src, tgt = self.__cat(src, tgt, src_scores, tgt_scores, src_saliency, tgt_saliency)

        # Decoder blocks

        src, tgt, src_coords, tgt_coords = self.nearest_up1(src, tgt, src_coords, tgt_coords, self.encoder_shortcut[2])
        src, tgt = self.conv1d_decoder1(src, tgt)
        src, tgt, src_coords, tgt_coords = self.nearest_up2(src, tgt, src_coords, tgt_coords, self.encoder_shortcut[1])
        src, tgt = self.conv1d_decoder2(src, tgt)
        src, tgt, src_coords, tgt_coords = self.nearest_up3(src, tgt, src_coords, tgt_coords, self.encoder_shortcut[0])
        src, tgt = self.conv1d_decoder3(src, tgt)

        print("success")
        return src, tgt, src_coords, tgt_coords, src_prob, tgt_prob, src_gnn_coords, tgt_gnn_coords

    def __shortcut_features(self, src, tgt, src_coords, tgt_coords):
        # divide the source and target features by 2 to make shortcut
        self.encoder_shortcut.append([src.clone().detach(), tgt.clone().detach(), src_coords.clone().detach(), tgt_coords.clone().detach()])

        return src, tgt

    def __scores(self, src, tgt):
        self.conv1d = Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        src_scores, tgt_scores = self.conv1d(src, tgt)

        return src_scores, tgt_scores

    def __scores_saliency(self, src, tgt, src_scores, tgt_scores):
        # N: the number of src points
        N, C = src.size()

        feats_cat = torch.cat((src, tgt), dim=0)
        feats_norm = tnf.normalize(feats_cat, p=2, dim=1)  # [N, C]

        # normalized points
        src_norm = feats_norm[:N, :]
        tgt_norm = feats_norm[N:, :]

        inner_products = torch.matmul(src_norm, tgt_norm.transpose(0, 1))
        temp = torch.exp(self.epsilon) + 0.03

        # saliency
        src_saliency = torch.matmul(tnf.softmax(inner_products / temp, dim=1), tgt_scores)
        tgt_saliency = torch.matmul(tnf.softmax(inner_products.permute(1, 0) / temp, dim=1), src_scores)

        return src_saliency, tgt_saliency

    @staticmethod
    def __cat(src, tgt, src_scores, tgt_scores, src_saliency, tgt_saliency):
        src = torch.cat((src, src_scores, src_saliency), dim=1)
        tgt = torch.cat((tgt, tgt_scores, tgt_saliency), dim=1)

        return src, tgt


if __name__ == '__main__':
    pass
