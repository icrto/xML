from torch import nn
import torch
from ConvMod import ConvMod


class Explainer(nn.Module):
    """ Class Explainer
    """

    def __init__(self, in_channels=3, exp_conv_filter_size=(3, 3), exp_pool_size=2, img_size=(224, 224), init_bias=10.0):

        super(Explainer, self).__init__()
        self.in_channels = in_channels
        self.exp_conv_filter_size = exp_conv_filter_size
        self.exp_pool_size = exp_pool_size
        self.img_size = img_size
        self.init_bias = init_bias

        self.conv_mod0 = ConvMod(self.in_channels, 32,
                                 self.exp_conv_filter_size)
        self.pool0 = nn.MaxPool2d(self.exp_pool_size)

        self.conv_mod1 = ConvMod(
            32, 64, self.exp_conv_filter_size)
        self.pool1 = nn.MaxPool2d(self.exp_pool_size)

        self.conv_mod2 = ConvMod(
            64, 128, self.exp_conv_filter_size)
        self.pool2 = nn.MaxPool2d(self.exp_pool_size)

        self.tr_conv0 = nn.ConvTranspose2d(
            128, 128, kernel_size=self.exp_pool_size, stride=self.exp_pool_size)
        self.conv_mod3 = ConvMod(
            128, 128, self.exp_conv_filter_size)

        self.tr_conv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=self.exp_pool_size, stride=self.exp_pool_size)
        self.conv_mod4 = ConvMod(
            64, 64, self.exp_conv_filter_size)

        self.tr_conv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=self.exp_pool_size, stride=self.exp_pool_size)
        self.conv_mod5 = ConvMod(
            32, 32, self.exp_conv_filter_size)

        self.conv_1x1 = nn.Conv2d(32, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ Initialises batch norm layer's bias to a pre-defined value.
            Guarantees that the initially produced explanations are white images (filled with ones) --> see paper for details.

        Arguments:
            m {torch.nn.Layer} -- layer in which we want to modify the initial bias value
        """
        if(type(m) == nn.BatchNorm2d):
            m.bias.data.fill_(self.init_bias)

    def forward(self, x):
        # stores the last convolutional layer for each level (to later add its resulting feature map to the corresponding upsampling layer as done in the original u-net)
        last_conv_per_level = []
        x = self.conv_mod0(x)
        last_conv_per_level.append(x)
        x = self.pool0(x)

        x = self.conv_mod1(x)
        last_conv_per_level.append(x)
        x = self.pool1(x)

        x = self.conv_mod2(x)
        last_conv_per_level.append(x)
        x = self.pool2(x)

        x = self.tr_conv0(x)
        x = torch.add(x, last_conv_per_level[2])
        x = self.conv_mod3(x)

        x = self.tr_conv1(x)
        x = torch.add(x, last_conv_per_level[1])
        x = self.conv_mod4(x)

        x = self.tr_conv2(x)
        x = torch.add(x, last_conv_per_level[0])
        x = self.conv_mod5(x)

        x = self.conv_1x1(x)
        x = self.bn(x)
        x = self.tanh(x)
        x = self.relu(x)

        return x
