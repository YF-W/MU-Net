import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

"""
MU-Net: Multi-kernels U-NET
Proposed By Yuefei Wang
Chengdu University
2023.12
This code is for free trial by fellow researchers, commercial use is strictly prohibited.
"""

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):              
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)           # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)          # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, idex, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.image_size = image_size


        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height ==0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.channel = num_patches + 1


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(self.channel, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.up = nn.Sequential()
        for i in range(idex):
            self.up.append(nn.ConvTranspose2d(self.channel, self.channel, kernel_size=2, stride=2))


    def forward(self, img):
        x = self.to_patch_embedding(img)        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape           # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)  
        x = torch.cat((cls_tokens, x), dim=1)               # 将cls_token拼接到patch token中去       (b, 65, dim)
        x += self.pos_embedding[:, :(n+1)]                  # 加位置嵌入（直接加）      (b, 65, dim)
        x = self.dropout(x)

        x = self.transformer(x)                                                 # (b, 65, dim)
        # print(x.shape)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)

        # x = self.to_latent(x)                                                   # Identity (b, dim)
        # return x
        # x = self.mlp_head(x)                                                 #  (b, num_classes)
        x = x.view(4, self.channel, 14, 14)
        x = self.up(x)
        
        return self.conv_1x1(x)



class AttentionBlcok(nn.Module):
    def __init__(
            self, in_c, image_size, patch_size, num_classes, dim, idex,channels=64,ReductionRatio=[4, 8, 16]
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
        self.vit = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = dim,
        channels=channels,
        idex = idex,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )


    def forward(self, x):
        x_orinal = x
        b, c, _, _ = x.size()

        y_1 = self.avgPool(x).view(b, c)
        y_1 = self.fc_1(y_1).view(b, c, 1, 1)

        y_2 = self.avgPool(x).view(b, c)
        y_2 = self.fc_2(y_2).view(b, c, 1, 1)

        y_3 = self.avgPool(x).view(b, c)
        y_3 = self.fc_3(y_3).view(b, c, 1, 1)

        z = (y_1 + y_2 + y_3)/3
        z = self.Sigmoid(z)

        x = self.vit(x)

        x = x * z

        return x + x_orinal

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
        self.Conv_Residuals = nn.ModuleList() #卷积存储池

        self.Conv_Residuals = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, bias=False)
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
        x_add = x_3 + x_5 + x_7 + x_9 + x
        x_bout1 = self.ReLu(self.BN_norblock(x_add))

        x_Residuals = self.Conv_Residuals(x_bout1)
        x_3 = self.ConvPool_lastBlock[0](x_bout1)
        x_5 = self.ConvPool_lastBlock[1](x_bout1)
        x_7 = self.ConvPool_lastBlock[2](x_bout1)
        x_9 = self.ConvPool_lastBlock[3](x_bout1)
        x_add = x_3 + x_5 + x_7 + x_9 + x_Residuals

        x_bout2 = self.ReLu(self.BN_lastblcok(x_add))

        return x_bout2



class Conv_1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class LayerConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(LayerConv, self).__init__()
        self.ConvBlock = ConvBlock(in_c, out_c)

    def forward(self, x):
        return self.ConvBlock(x)

class Aided_encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=11, dilation=11, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=9, dilation=9, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_9x9 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=7, dilation=7, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return F.relu(self.conv_1x1(x) + self.conv(x) + self.conv_5x5(x) + self.conv_9x9(x))

class MUNet_704(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(MUNet_704, self).__init__()
        self.ups = nn.ModuleList()  # 将多个Module加入list，但不存在实质性顺序，参考python的list
        self.downs = nn.ModuleList()
        self.downs_1x1 = nn.ModuleList()
        self.downs_Aided = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Dynamic_Pooling = nn.ModuleList()

        self.Dynamic_Pooling.append(nn.AdaptiveAvgPool2d((int(224/1),int(224/1))))
        self.Dynamic_Pooling.append(nn.AdaptiveAvgPool2d((int(224/2),int(224/2))))
        self.Dynamic_Pooling.append(nn.AdaptiveAvgPool2d((int(224/4),int(224/4))))
        self.Dynamic_Pooling.append(nn.AdaptiveAvgPool2d((int(224/8),int(224/8))))

        # Down part of MyUNET
        for feature in features:
            self.downs.append(LayerConv(in_channels, out_c=feature))
            self.downs_Aided.append(Aided_encoder(in_channels, feature))
            self.downs_1x1.append(Conv_1x1(feature * 2, feature))
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
        self.final_conv = nn.Conv2d(960, out_channels, kernel_size=1)

        self.layer1_SkipAttention = AttentionBlcok(in_c=64,image_size=224, patch_size=16, num_classes=224 * 224, dim=196, channels=64,idex = 4)
        self.layer2_SkipAttention = AttentionBlcok(in_c=128, image_size=112, patch_size=8, num_classes=112 * 112, dim=196, channels=128,idex = 3)
        self.layer3_SkipAttention = AttentionBlcok(in_c=256, image_size=56, patch_size=4, num_classes=56 * 56, dim=196, channels=256, idex = 2)
        self.layer4_SkipAttention = AttentionBlcok(in_c=512, image_size=28, patch_size=2, num_classes=28 * 28, dim=196, channels=512, idex = 1)

        self.bottlen_to_decoder = nn.ModuleList()
        self.decoder_to_out = nn.ModuleList()
        self.decoder_to_out.append(nn.Upsample(scale_factor=8))
        self.decoder_to_out.append(nn.Upsample(scale_factor=4))
        self.decoder_to_out.append(nn.Upsample(scale_factor=2))
        self.decoder_to_out.append(nn.Upsample(scale_factor=1))
        for feature in reversed(features):
            self.bottlen_to_decoder.append(nn.Upsample(scale_factor=2))
            self.bottlen_to_decoder.append(Conv_1x1(feature + 1024, feature))

    def forward(self, x):
        skip_connections = []
        decoder = []

        # decoder part
        pool_idx = 0
        x_origin = x
        for idex in range(len(self.downs)):
            x_origin = self.downs_Aided[idex](x_origin)
            x = torch.cat([self.Dynamic_Pooling[pool_idx](x_origin), self.downs[idex](x)], dim=1)
            pool_idx += 1
            x = self.downs_1x1[idex](x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        x_bottlen = x

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
            # w = skip_connections_attens[idx//2][0]
            # x = x * w
            if x.shape != skip_connection_atten.shape:
                x = TF.resize(x, size=skip_connection_atten.shape[2:])

            concat_skip = torch.cat([skip_connection_atten, x], dim=1)
            x = self.ups[idx+1](concat_skip)
            x_bottlen = self.bottlen_to_decoder[idx](x_bottlen)
            x = torch.cat([x_bottlen, x], dim=1)
            x = self.bottlen_to_decoder[idx + 1](x)
            x_1 = self.decoder_to_out[idx//2](x)
            decoder.append(x_1)
        x = torch.cat([decoder[0], decoder[1], decoder[2], decoder[3]], dim=1)

        return self.final_conv(x)



def test():
    x = torch.randn((4, 3, 224, 224))
    model = MUNet_704(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    # assert preds.shape == x.shape

if __name__ == "__main__":
    test()

# x = torch.randn((4, 128, 112, 112))
# module = ViT(
#         image_size = 112,
#         patch_size = 8,
#         num_classes = 112 * 112,
#         dim = 512,
#         channels=128,
#         depth = 6,
#         heads = 16,
#         mlp_dim = 2048,
#         dropout = 0.1,
#         emb_dropout = 0.1
#     )
# pre = module(x)
# print(pre.shape)
