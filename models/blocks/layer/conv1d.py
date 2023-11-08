import torch.nn as nn


class Conv1d(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=1, bias=True):
        super().__init__()

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, stride=stride)

    def forward(self, src, tgt):
        src = src.permute(1, 0)
        tgt = tgt.permute(1, 0)
        src = self.conv1d(src)
        tgt = self.conv1d(tgt)
        src = src.permute(1, 0)
        tgt = tgt.permute(1, 0)

        return src, tgt


if __name__ == '__main__':
    pass
