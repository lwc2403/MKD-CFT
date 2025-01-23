import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
    
class Residual(nn.Module):  
    def __init__(self, i_c, num_c,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(i_c, num_c,
                               kernel_size=5, padding=2, stride=strides)
        self.conv2 = nn.Conv2d(num_c, num_c,
                               kernel_size=5, padding=2)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(i_c, num_c,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_c)
        self.bn2 = nn.BatchNorm2d(num_c)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
def resnet_block(i_c, num_c, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(i_c, num_c,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_c, num_c))
    return blk
class Res_net_2d(nn.Module):
    def __init__(self, args):
        super(Res_net_2d, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.Res_net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten())

    def forward(self, x):
        x = self.Res_net(x)
        return x
    
    
class ResidualBlock(nn.Module):
    def __init__(self, i_d, o_c, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(i_d, o_c, kernel_size=5,
            stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=o_c)
        self.conv2 = nn.Conv1d(o_c, o_c, kernel_size=5,
            stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=o_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or i_d != o_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(i_d, o_c, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm1d(o_c))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(torch.nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.layers = 6
        self.i_d = 1024
        self.i_c = 64
        self.n_classes = 10
        self.hidden_sizes = [100] * self.layers
        self.num_blocks = [2] * self.layers
        self.conv1 = nn.Conv1d(1, self.i_c, kernel_size=5, stride=1,
            padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.i_c)
        layers = []
        strides = [1] + [2] * (len(self.hidden_sizes) - 1)
        for idx, hidden_size in enumerate(self.hidden_sizes):
            layers.append(self._make_layer(hidden_size, self.num_blocks[idx],
                stride=strides[idx]))
        self.encoder = nn.Sequential(*layers)
        self.z_dim = self._get_encoding_size()
    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        z = x.view(x.size(0), -1)
        return z
    def forward(self, x):
        z = self.encode(x)
        return z
    def _make_layer(self, o_c, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.i_c, o_c,
                stride=stride))
            self.i_c = o_c
        return nn.Sequential(*blocks)
    def _get_encoding_size(self):
        temp = Variable(torch.rand(1, 1, self.i_d))
        z = self.encode(temp)
        z_dim = z.data.size(1)
        return z_dim
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out
class ResNet18(nn.Module):
    def __init__(self,args):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.num_classes=10
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def _make_layer(self, block, out_channels, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    


