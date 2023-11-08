import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.mlp = nn.Linear(in_channel, out_channel)
        # Sett the bias of the last mlp layer equal to zero.
        nn.init.constant_(self.mlp.bias, 0.0)

    def forward(self, src, tgt):
        # Source features
        src = self.mlp(src)

        # target features
        tgt = self.mlp(tgt)

        return src, tgt


if __name__ == "__main__":
    pass
