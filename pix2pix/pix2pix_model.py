import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    # def __init__(self, in_channels=64, features=[256, 512, 1024, 2048]):
    def __init__(self, in_channels=84, features=[336, 672, 1344, 2688]):
        super().__init__()
        self.initial = nn.Sequential(
            # nn.AvgPool3d((10, 1, 1), stride=(10, 1, 1)),
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CnnBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # p = nn.AvgPool3d((13, 1, 1), stride=(13, 1, 1))
        # x = p(x)
        # y = p(y)
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):

    # def __init__(self, in_channels=64, features=256):
    def __init__(self, in_channels=84, features=336):
        super().__init__()
        self.initial_down = nn.Sequential(
            # nn.AvgPool3d((10, 1, 1), stride=(10, 1, 1)),
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU()
        )

        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()

        )

        # self.up1 = Block(features * 8, features * 8, down=False, act="leaky", use_dropout=True)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(features * 8, features * 8, 4, 3, 1, bias=False),
                                 nn.BatchNorm2d(features * 8),
                                 nn.ReLU())  # if act == "relu" else nn.LeakyReLU(0.2))

        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="leaky", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="leaky", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="leaky", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="leaky", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="leaky", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act="leaky", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        # print('d1 shape: ', d1.shape)
        d2 = self.down1(d1)
        # print('d2 shape: ', d2.shape)
        d3 = self.down2(d2)
        # print('d3 shape: ', d3.shape)
        d4 = self.down3(d3)
        # print('d4 shape: ', d4.shape)
        d5 = self.down4(d4)
        # print('d5 shape: ', d5.shape)
        d6 = self.down5(d5)
        # print('d6 shape: ', d6.shape)
        d7 = self.down6(d6)
        # print('d7 shape: ', d7.shape)

        bottleneck = self.bottleneck(d7)
        # print('bottleneck shape: ', bottleneck.shape)

        up1 = self.up1(bottleneck)
        # print('up1 shape: ', up1.shape)
        up2 = self.up2(torch.cat([up1, d7], 1))
        # print('up2 shape: ', up2.shape)
        up3 = self.up3(torch.cat([up2, d6], 1))
        # print('up3 shape: ', up3.shape)
        up4 = self.up4(torch.cat([up3, d5], 1))
        # print('up4 shape: ', up4.shape)
        up5 = self.up5(torch.cat([up4, d4], 1))
        # print('up5 shape: ', up5.shape)
        up6 = self.up6(torch.cat([up5, d3], 1))
        # print('up6 shape: ', up6.shape)
        up7 = self.up7(torch.cat([up6, d2], 1))
        # print('up7 shape: ', up7.shape)
        # print('final up shape: ', torch.cat([up7, d1], 1).shape)

        return self.final_up(torch.cat([up7, d1], 1))


def dis_test():
    print('disc test')
    x = torch.rand((1, 84, 640, 640))
    y = torch.rand((1, 84, 640, 640))
    model = Discriminator()
    pred = model(x, y)
    print(pred.shape)


def gen_test():
    print('gen test')
    x = torch.rand((1, 84, 320, 320))
    print(x.shape)
    model = Generator()
    pred = model(x)
    print(pred.shape)

#
if __name__ == '__main__':
    dis_test()
    gen_test()
