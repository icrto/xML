import torch
from torch import nn
from ConvMod import ConvMod
class VGGClf(nn.Module):
    def __init__(self, in_channels=3, dec_conv_filter_size=(3, 3), dec_pool_size=2, img_size=(224, 224), dropout=0.3, num_classes=2):

        super(VGGClf, self).__init__()
        self.in_channels = in_channels
        self.dec_conv_filter_size = dec_conv_filter_size
        self.dec_pool_size = dec_pool_size
        self.dropout = dropout
        self.img_size = img_size
        self.num_classes = num_classes
        
        self.conv_mod0 = ConvMod(self.in_channels, 32, self.dec_conv_filter_size)
        self.pool0 = nn.MaxPool2d(self.dec_pool_size)

        self.conv_mod1 = ConvMod(32, 64, self.dec_conv_filter_size)
        self.pool1 = nn.MaxPool2d(self.dec_pool_size)

        self.conv_mod2 = ConvMod(64, 128, self.dec_conv_filter_size)
        self.pool2 = nn.MaxPool2d(self.dec_pool_size)

        self.conv_mod3 = ConvMod(128, 256, self.dec_conv_filter_size)
        self.pool3 = nn.MaxPool2d(self.dec_pool_size)
        self.global_pool = nn.MaxPool2d((14,14))

        self.dense0 = nn.Linear(256, 128)
        self.actv0 = nn.Sigmoid()
        self.dropout0 = nn.Dropout(p=self.dropout)
        self.dense1 = nn.Linear(128, 128)
        self.actv1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=self.dropout)

        self.last_dense = nn.Linear(128, self.num_classes)
    
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if(type(m) == ConvMod):
            m.apply(m._init_weights_clf)
        elif(type(m) == nn.Linear):
            m.bias.data.fill_(0.0) 
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x, expl):
        last_expl = expl
        x = self.conv_mod0(x)
        x = torch.mul(x, torch.cat([last_expl] * 32, dim=1))
        x = self.pool0(x)
        last_expl = self.pool0(last_expl)

        x = self.conv_mod1(x)
        x = torch.mul(x, torch.cat([last_expl] * 64, dim=1))
        x = self.pool1(x)
        last_expl = self.pool1(last_expl)

        x = self.conv_mod2(x)
        x = torch.mul(x, torch.cat([last_expl] * 128, dim=1))
        x = self.pool2(x)
        last_expl = self.pool2(last_expl)

        x = self.conv_mod3(x)
        x = torch.mul(x, torch.cat([last_expl] * 256, dim=1))
        x = self.pool3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)      
        x = self.dense0(x)
        x = self.dropout0(self.actv0(x))
       
        x = self.dense1(x)
        x = self.dropout1(self.actv1(x))

        x = self.last_dense(x)
       
        return x

    


          