import torch
import torch.nn.functional as F
import torch.nn as nn
from toolbox.loss.kd_loss import *
import math

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GCN(nn.Module):
    def __init__(self, channel, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.conv2(self.relu(h))
        return h

class Region_Unit(nn.Module):
    def __init__(self, channelin, size, ConvNd=nn.Conv2d):
        super(Region_Unit, self).__init__()
        if (channelin > 320):
            self.InterChannel = channelin // 16
        else:
            self.InterChannel = channelin // 2

        if(channelin == 6):
            self.InterChannel = channelin

        if (size > 16):
            self.region_size1 = 3
            self.region_size2 = 5
            self.region_size3 = 7
        else:
            self.region_size1 = 1
            self.region_size2 = 2
            self.region_size3 = 3
        self.stride1 = self.region_size1
        self.stride2 = math.ceil(self.region_size2 / 2)
        self.stride3 = math.ceil(self.region_size3 / 2)
        if(size == 256):
            self.region_size1 = 5
            self.region_size2 = 7
            self.region_size3 = 9
            self.stride2 = math.ceil(self.region_size2 / 2)
            self.stride3 = math.ceil(self.region_size3 / 2)

        self.num_nodes1 = (size - self.region_size1) // self.stride1 + 1
        self.num_nodes2 = (size - self.region_size2) // self.stride2 + 1
        self.num_nodes3 = (size - self.region_size3) // self.stride3 + 1

        # reduce dim
        self.conv_reduce = ConvNd(channelin, self.InterChannel, kernel_size=1)
        # projection map
        self.conv_proj1 = ConvNd(self.InterChannel, self.InterChannel, kernel_size=self.region_size1,
                                 stride=self.stride1)
        self.conv_proj2 = ConvNd(self.InterChannel, self.InterChannel, kernel_size=self.region_size2,
                                 stride=self.stride2)
        self.conv_proj3 = ConvNd(self.InterChannel, self.InterChannel, kernel_size=self.region_size3,
                                 stride=self.stride3)

        # reasoning via graph convolution
        self.gcn1 = GCN(self.InterChannel, num_node=self.num_nodes1 * self.num_nodes1)
        self.gcn2 = GCN(self.InterChannel, num_node=self.num_nodes2 * self.num_nodes2)
        self.gcn3 = GCN(self.InterChannel, num_node=self.num_nodes3 * self.num_nodes3)

    def forward(self, x):
        b, c, h, w = x.size()
        x_reduce = self.conv_reduce(x)
        # projection map
        x_proj1 = self.conv_proj1(x_reduce)
        x_proj2 = self.conv_proj2(x_reduce)
        x_proj3 = self.conv_proj3(x_reduce)
        # print(x_proj1.shape, x_proj2.shape, x_proj3.shape)

        x_proj1 = torch.matmul(x_proj1, x_proj1.permute(0, 1, 3, 2)).view(b, self.InterChannel, -1)
        x_proj2 = torch.matmul(x_proj2, x_proj2.permute(0, 1, 3, 2)).view(b, self.InterChannel, -1)
        x_proj3 = torch.matmul(x_proj3, x_proj3.permute(0, 1, 3, 2)).view(b, self.InterChannel, -1)

        # graph based reasoning
        x_proj1 = F.softmax(self.gcn1(x_proj1), dim=1).view(b, self.InterChannel, self.num_nodes1, self.num_nodes1)
        x_proj2 = F.softmax(self.gcn2(x_proj2), dim=1).view(b, self.InterChannel, self.num_nodes2, self.num_nodes2)
        x_proj3 = F.softmax(self.gcn3(x_proj3), dim=1).view(b, self.InterChannel, self.num_nodes3, self.num_nodes3)

        return x_proj1, x_proj2, x_proj3

class SDGC(nn.Module):
    def __init__(self, channelT, sizeT, channelS, sizeS):
        super(SDGC, self).__init__()
        self.convDownT = BasicConv2d(channelT, channelS, kernel_size=1)
        self.regionT = Region_Unit(channelS, sizeT)
        self.regionS = Region_Unit(channelS, sizeS)
        if (channelS > 320):
            self.InterChannel = channelS // 16
        else:
            self.InterChannel = channelS // 2

        if(channelS == 6):
            self.InterChannel = channelS

        self.vidloss = VID(self.InterChannel)

    def forward(self, teacher, student):
        b, c, h, w = student.size()
        if(c != 6):
            teacher = self.convDownT(teacher)
        Tmap1, Tmap2, Tmap3 = self.regionT(teacher)
        Smap1, Smap2, Smap3 = self.regionS(student)
        loss = self.vidloss(Tmap1, Smap1) + self.vidloss(Tmap2, Smap2) + self.vidloss(Tmap3, Smap3)
        return loss

class CBAM(nn.Module):
    def __init__(self, in_channel, kernel_size=7):
        super(CBAM, self).__init__()
        if (in_channel <= 320):
            self.interChannel = in_channel // 2
        else:
            self.interChannel = in_channel // 16
        # 通道注意力机制
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=self.interChannel, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=self.interChannel, out_features=in_channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # 空间注意力机制
        self.conv = BasicConv2d(in_planes=2, out_planes=1, kernel_size=kernel_size, padding= kernel_size // 2)

    def forward(self, x):
        # 通道注意力机制
        maxout = self.max_pool(x)
        maxout = self.mlp(maxout.view(maxout.size(0), -1))
        avgout = self.avg_pool(x)
        avgout = self.mlp(avgout.view(avgout.size(0), -1))
        channel_weight = self.sigmoid(maxout + avgout)
        channel_weight = channel_weight.view(x.size(0), x.size(1), 1, 1)
        channel_out = channel_weight * x
        # 空间注意力机制
        max_out, _ = torch.max(channel_out, dim=1, keepdim=True)
        mean_out = torch.mean(channel_out, dim=1, keepdim=True)
        out = torch.cat((max_out, mean_out), dim=1)
        spatial_weight = self.sigmoid(self.conv(out))
        return spatial_weight

class PEM(nn.Module):
    def __init__(self, channelin):
        super(PEM, self).__init__()
        self.conv = BasicConv2d(channelin, 1, kernel_size=1)
        self.cbam = CBAM(channelin)

    def forward(self, feature):
        # global mask
        theta_feature = self.conv(feature) # [b, 1, h, w]
        phi_feature = feature.permute(0, 1, 3, 2)  # [b, 1, w, h]
        global_mask = F.softmax(torch.matmul(theta_feature, phi_feature), dim=1) # [b, 1, h, w]
        global_feature = global_mask * feature + feature

        # local mask
        local_mask = self.cbam(global_feature)

        mapPEM = local_mask
        return mapPEM

class SKD(nn.Module):
    def __init__(self, channelT, channelS, channelSH):
        super(SKD, self).__init__()
        self.convDownT = BasicConv2d(channelT, channelS, kernel_size=1)
        self.convDownSH = BasicConv2d(channelSH, channelS, kernel_size=1)
        self.pem_T = PEM(channelS)
        self.pem_S = PEM(channelS)
        self.pem_SH = PEM(channelS)

        self.vidmap = VID(1)
        self.vidfeature = VID(channelS)

    def forward(self, featureT, featureS, featureSH):
        b, c, h, w = featureS.shape
        featureSH = self.convDownSH(F.interpolate(featureSH, (h, w), mode='bilinear', align_corners=True))
        featureT = self.convDownT(featureT)
        Tmap = self.pem_T(featureT)
        SHmap = self.pem_SH(featureSH)
        featureSEnhenced = featureS * (Tmap + SHmap)
        Smap = self.pem_S(featureSEnhenced)

        loss = self.vidmap(Smap, SHmap) + self.vidmap(Smap, Tmap) + self.vidfeature(featureS * Smap, featureT * Tmap) + self.vidfeature(featureS * Smap, featureSH * SHmap)
        return loss