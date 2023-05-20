import torch
import torch.nn as nn

# 1x1, 3x3のConvolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )

class Bottleneck(nn.Module):
    expansion = 4  # チャネル数を何倍にするか

    def __init__(
        self,
        in_channels,  # 入力チャネル
        channels,  # 中間層チャネル -> 出力チャネル: self.expansion * channels
        stride=1
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            conv1x1(in_channels, channels),
            nn.BatchNorm2d(channels)
        )
        self.conv2 = nn.Sequential(
            conv3x3(channels, channels, stride),
            nn.BatchNorm2d(channels)
        )
        self.conv3 = nn.Sequential(
            conv1x1(channels, self.expansion * channels),
            nn.BatchNorm2d(self.expansion * channels)
        )
        self.relu = nn.ReLU()

        # 畳み込んだものと元のxのチャネル数が異なる場合
        # down sampling(conv1x1を使った線型変換による形状一致)を行う
        if in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, self.expansion * channels, stride),
                nn.BatchNorm2d(self.expansion * channels)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.relu(out)
        
        out = out + identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block,  # residual block
        layers,  # 各layerの繰り返し回数
        num_classes=10  # 分類クラス数
    ):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # poolingでダウンサンプリング
            self._make_layer(block, 64, layers[0], stride=1)
        )
        # 以降は畳み込みでダウンサンプリング
        self.conv3 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block.expansion * 512, num_classes)

        # 重みを初期化する。He initialization (torch.nn.init.kaimingnormal) を使用。
        # Batch Normalization 層の初期化は重み1、バイアス0で初期化。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride):
        layers = []

        # 最初のblockは，
        # self.in_channels != block.expansion * channels
        # の可能性があるため別に行う
        layers.append(block(self.in_channels, channels, stride))  # ダウンサンプリングする場合，stride=2となっている

        # 以降出力がblock.expansion * channelsになるので更新しておく
        self.in_channels = block.expansion * channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))  # ここからはstride=1

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])