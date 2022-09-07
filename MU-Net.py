import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
"""
MU-Net
With Multi-scale Convolution Block
With Multi-scale SEBlock
With Depthwise Separable Convolution
"""
class AttentionBlcok(nn.Module):
    def __init__(
            self, in_c, ReductionRatio=[4, 8, 16]
    ):
        super(AttentionBlcok, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear(in_c, in_c//ReductionRatio[0], bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c//ReductionRatio[0], in_c, bias=False),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_c, in_c//ReductionRatio[1], bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c//ReductionRatio[1], in_c, bias=False),
        )       
        self.fc_3 = nn.Sequential(
            nn.Linear(in_c, in_c//ReductionRatio[2], bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c//ReductionRatio[2], in_c, bias=False),
        )
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        y_1 = self.avgPool(x).view(b, c)
        y_1 = self.fc_1(y_1).view(b, c, 1, 1)

        y_2 = self.avgPool(x).view(b, c)
        y_2 = self.fc_2(y_2).view(b, c, 1, 1)

        y_3 = self.avgPool(x).view(b, c)
        y_3 = self.fc_3(y_3).view(b, c, 1, 1)

        z = (y_1 + y_2 + y_3)/3
        z = self.Sigmoid(z)

        z = x * z.expand_as(x)
        return z    


# class DoubleConv(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(  # 
#             nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),  # inplace 是否进行覆盖运算
#             nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)

class DSCConv(nn.Module):
    def __init__(self,in_c,out_c):
        super(DSCConv, self).__init__()
        
        self.depth_conv = nn.Conv2d(in_channels=in_c,
                                    out_channels=in_c,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_c)
        
        self.point_conv = nn.Conv2d(in_channels=in_c,
                                    out_channels=out_c,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()
        self.ConvPool_lastBlock = nn.ModuleList() #卷积存储池

        self.ConvPool_lastBlock.append((nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3,  padding=1, bias=False)))
        self.ConvPool_lastBlock.append((nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=5,  padding=2, bias=False)))
        self.ConvPool_lastBlock.append((nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=7,  padding=3, bias=False)))
        self.ConvPool_lastBlock.append((nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=9,  padding=4, bias=False)))
        
        self.ConvPool_norblock = nn.ModuleList()

        self.ConvPool_norblock.append((nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3,  padding=1, bias=False)))
        self.ConvPool_norblock.append((nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=5,  padding=2, bias=False)))
        self.ConvPool_norblock.append((nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=7,  padding=3, bias=False)))
        self.ConvPool_norblock.append((nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=9,  padding=4, bias=False)))

        self.BN_norblock = nn.BatchNorm2d(in_c)
        self.BN_lastblcok = nn.BatchNorm2d(out_c)
        self.ReLu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_3 = self.ConvPool_norblock[0](x)
        x_5 = self.ConvPool_norblock[1](x)
        x_7 = self.ConvPool_norblock[2](x)
        x_9 = self.ConvPool_norblock[3](x)
        x_add = x_3 + x_5 + x_7 + x_9
        x_bout1 = self.ReLu(self.BN_norblock(x_add))

        x_3 = self.ConvPool_lastBlock[0](x_bout1)
        x_5 = self.ConvPool_lastBlock[1](x_bout1)
        x_7 = self.ConvPool_lastBlock[2](x_bout1)
        x_9 = self.ConvPool_lastBlock[3](x_bout1)
        x_add = x_3 + x_5 + x_7 + x_9
        
        x_bout2 = self.ReLu(self.BN_lastblcok(x_add))
        
        return x_bout2
        




class LayerConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(LayerConv, self).__init__()
        self.ConvBlock = ConvBlock(in_c, out_c)

    def forward(self, x):
        return self.ConvBlock(x)


class MyUNET(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256, 512],
    ):
        super(MyUNET, self).__init__()
        self.ups = nn.ModuleList()  # 将多个Module加入list，但不存在实质性顺序，参考python的list
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Dynamic_Pooling = nn.ModuleList()

        self.Dynamic_Pooling.append(nn.AdaptiveAvgPool2d((int(512/1),int(512/1))))
        self.Dynamic_Pooling.append(nn.AdaptiveAvgPool2d((int(512/2),int(512/2))))
        self.Dynamic_Pooling.append(nn.AdaptiveAvgPool2d((int(512/4),int(512/4))))
        self.Dynamic_Pooling.append(nn.AdaptiveAvgPool2d((int(512/8),int(512/8))))

        # Down part of MyUNET
        for feature in features:
            self.downs.append(LayerConv(in_channels+3, out_c=feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DSCConv(feature*2, feature))

        self.bottleneck = DSCConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.layer1_SkipAttention = AttentionBlcok(in_c=64)
        self.layer2_SkipAttention = AttentionBlcok(in_c=128)
        self.layer3_SkipAttention = AttentionBlcok(in_c=256)
        self.layer4_SkipAttention = AttentionBlcok(in_c=512)

    def forward(self, x):
        skip_connections = []

        # decoder part
        pool_idx = 0
        x_origin = x
        for down in self.downs:
            x = torch.cat((self.Dynamic_Pooling[pool_idx](x_origin), x), dim=1)
            pool_idx += 1
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections_attens = []
        skip_connections_attens.append(self.layer1_SkipAttention(skip_connections[0]))
        skip_connections_attens.append(self.layer2_SkipAttention(skip_connections[1]))
        skip_connections_attens.append(self.layer3_SkipAttention(skip_connections[2]))
        skip_connections_attens.append(self.layer4_SkipAttention(skip_connections[3]))

        skip_connections_attens = skip_connections_attens[::-1]

        # encoder part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection_atten = skip_connections_attens[idx//2]

            if x.shape != skip_connection_atten.shape:
                x = TF.resize(x, size=skip_connection_atten.shape[2:])

            concat_skip = torch.cat((skip_connection_atten, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((4, 3, 512, 512))
    model = MyUNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()


