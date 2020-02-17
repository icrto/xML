from torch import nn
class ConvMod(nn.Module):
    def __init__(self, in_channels=3, nfilters=32, conv_filter_size=(3, 3), init_bias=10.0):
        super(ConvMod, self).__init__()
        self.in_channels = in_channels
        self.nfilters = nfilters
        self.conv_filter_size = conv_filter_size
        self.init_bias = init_bias

        self.conv0 = nn.Conv2d(in_channels, self.nfilters, self.conv_filter_size, padding=1, padding_mode='zeros')
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(self.nfilters, self.nfilters, self.conv_filter_size, padding=1, padding_mode='zeros')
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.relu0(self.conv0(x))
        x = self.relu1(self.conv1(x))
        return x

    def _init_weights(self, m):
        if(type(m) == nn.Conv2d):
            m.bias.data.fill_(self.init_bias) 
                 
    def _init_weights_clf(self, m):
        if(type(m) == nn.Conv2d):
            m.bias.data.fill_(0.0) 
            nn.init.xavier_uniform_(m.weight)
