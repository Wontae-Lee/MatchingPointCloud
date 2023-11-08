import torch.nn as nn


class LeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.1):
        super().__init__()

        self.leak_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, src, tgt):
        # source features
        src = self.leak_relu(src)

        # target features
        tgt = self.leak_relu(tgt)

        return src, tgt


if __name__ == "__main__":
    pass
