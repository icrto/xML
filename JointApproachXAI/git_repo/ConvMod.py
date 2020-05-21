from torch import nn


class ConvMod(nn.Module):
    """
        Class ConvMod (used in Explainer and VGG).
        Consists in two conv-relu stages.
    """

    def __init__(self, in_channels=3, nfilters=32, conv_filter_size=(3, 3)):
        super(ConvMod, self).__init__()
        self.in_channels = in_channels
        self.nfilters = nfilters
        self.conv_filter_size = conv_filter_size

        self.conv0 = nn.Conv2d(in_channels, self.nfilters,
                               self.conv_filter_size, padding=1, padding_mode='zeros')
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(self.nfilters, self.nfilters,
                               self.conv_filter_size, padding=1, padding_mode='zeros')
        self.relu1 = nn.ReLU()

    def forward(self, x):

        x = self.relu0(self.conv0(x))
        x = self.relu1(self.conv1(x))
        return x
