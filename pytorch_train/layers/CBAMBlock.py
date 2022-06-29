import numpy as np
import torch
from torch import nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)

        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, fusion, a_forces, l_forces):

        #fusion 首先要经过一个卷积操作 这个操作是为了识别误差模式的，也就是为后续权重计算做基础模式，能得到我们所需要关注的区域


        copy_a = a_forces
        copy_l = l_forces

        # ads light 在channel的注意力 这里分开计算，得到在该误差模式下,在每个专家预测上关注哪些通道
        a_att_c =self.ca(a_forces)
        l_att_c =self.ca(l_forces)

        fusion_a = fusion * a_att_c
        fusion_l = fusion * l_att_c

        a_forces = a_att_c * a_forces
        l_forces = l_att_c * l_att_c

        a_att_s = self.sa(a_forces)
        l_att_s = self.sa(l_forces)

        fusion_a = fusion_a * a_att_s
        fusion_l = fusion_l * l_att_s

        fusion_a = fusion_a.unsqueeze(dim=0)
        fusion_l = fusion_l.unsqueeze(dim=0)


        expert_weight = torch.softmax(torch.cat([fusion_a,fusion_l],dim=0), dim=0)

        a_forces = expert_weight[0].unsqueeze(dim=0) * copy_a

        l_forces = expert_weight[1].unsqueeze(dim=0) * copy_l


        return a_forces, l_forces


if __name__ == '__main__':
    # input = torch.randn(50, 512, 7, 7)
    # kernel_size = input.shape[2]
    # cbam = CBAMBlock(channel=512, reduction=16, kernel_size=kernel_size)
    # output = cbam(input)
    # print(output.shape)  # 经过CBAM注意力模块后，输入维度保持不变

    ads_put = torch.randn(3,64,30,30) * 0.1
    light_put = torch.randn(3, 64, 30, 30) * 0.1
    fusion = torch.randn(3, 64, 30, 30) * 0.1

    cbam = CBAMBlock(channel=64, reduction=16, kernel_size=10)

    out = cbam(fusion,ads_put,light_put)
    print(out.shape)

