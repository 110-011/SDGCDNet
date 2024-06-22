import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATConv
from torch_geometric.data import Data
import cv2
import numpy as np

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

class SCM(nn.Module):
    def __init__(self):
        super(SCM, self).__init__()
        self.conv = BasicConv2d(3, 1, kernel_size=1)
        self.conv3 = BasicConv2d(2, 1, kernel_size=3, padding=1)
        self.conv5 = BasicConv2d(2, 1, kernel_size=5, padding=2)
        self.conv7 = BasicConv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid_3 = nn.Sigmoid()
        self.sigmoid_5 = nn.Sigmoid()
        self.sigmoid_7 = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        w3 = self.sigmoid_3(self.conv3(x))
        w5 = self.sigmoid_5(self.conv5(x * w3))
        w7 = self.conv7(x * w5)
        return w7

class CCM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CCM, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.max_pool1 = nn.AdaptiveMaxPool2d(1)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(3)
        self.max_pool3 = nn.AdaptiveMaxPool2d(3)
        self.avg_pool5 = nn.AdaptiveAvgPool2d(5)
        self.max_pool5 = nn.AdaptiveMaxPool2d(5)
        self.sharedMLP1 = nn.Sequential(
            BasicConv2d(in_planes, in_planes // ratio, 1),
            BasicConv2d(in_planes // ratio, in_planes, 1),
        )
        self.sharedMLP3 = nn.Sequential(
            BasicConv2d(in_planes, in_planes // ratio, 1),
            BasicConv2d(in_planes // ratio, in_planes // ratio, kernel_size=3),
            BasicConv2d(in_planes // ratio, in_planes, kernel_size=1),
        )
        self.sharedMLP5 = nn.Sequential(
            BasicConv2d(in_planes, in_planes // ratio, 1),
            BasicConv2d(in_planes // ratio, in_planes // ratio, kernel_size=5),
            BasicConv2d(in_planes // ratio, in_planes, kernel_size=1),
        )
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid3 = nn.Sigmoid()
        self.sigmoid5 = nn.Sigmoid()
    def forward(self, x):
        w1 = self.sigmoid1(self.sharedMLP1(self.avg_pool1(x)) + self.sharedMLP1(self.max_pool1(x)))
        w3 = self.sigmoid3(self.sharedMLP3(self.avg_pool3(w1 * x)) + self.sharedMLP3(self.max_pool3(w1 * x)))
        w5 = self.sigmoid5(self.sharedMLP5(self.avg_pool5(w3 * x)) + self.sharedMLP5(self.max_pool5(w3 * x)))
        return w5

class ASFE(nn.Module):
    def __init__(self, channel):
        super(ASFE, self).__init__()
        self.channel = channel
        self.scm = SCM()
        self.ccm = CCM(channel)

    def forward(self, rgb, depth):
        scm_r = self.scm(rgb)
        scm_d = self.scm(depth * scm_r)
        ccm_r = self.ccm(rgb * scm_d)
        ccm_d = self.ccm(depth * scm_r * ccm_r)
        rgb = rgb * ccm_r * ccm_d + rgb
        depth = depth * ccm_d + depth
        return rgb, depth, ccm_d

class GCNOnFeatureMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNOnFeatureMap, self).__init__()
        self.conv1 = ChebConv(in_channels, out_channels, K=1)
        self.conv2 = ChebConv(out_channels, out_channels, K=3)
        self.conv3 = ChebConv(out_channels, out_channels, K=5)
        self.norm1 = torch.nn.BatchNorm1d(out_channels)
        self.norm2 = torch.nn.BatchNorm1d(out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 第一层图卷积
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        # 第二层图卷积
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        # 第三层图卷积
        x = self.conv3(x, edge_index)
        return x

def construct_graph_(feature_map):
    # 图的节点表示每个像素的特征，这里使用特征图中的值作为特征
    num_rows, num_cols = feature_map.shape[2], feature_map.shape[3]
    num_nodes = num_rows * num_cols
    x = feature_map.permute(0, 2, 3, 1).reshape(-1, 64)
    # 图的边表示像素之间的连接关系
    edge_index = []
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            # 添加右侧、下方、右下方、左侧、上方和左上方的相邻像素
            if j < num_cols - 1:  # 右侧
                edge_index.append([index, index + 1])
            if i < num_rows - 1:  # 下方
                edge_index.append([index, index + num_cols])
            if j < num_cols - 1 and i < num_rows - 1:  # 右下方
                edge_index.append([index, index + num_cols + 1])
            if j > 0:  # 左侧
                edge_index.append([index, index - 1])
            if i > 0:  # 上方
                edge_index.append([index, index - num_cols])
            if j > 0 and i > 0:  # 左上方
                edge_index.append([index, index - num_cols - 1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # 构造图数据
    data = Data(x=x, edge_index=edge_index)
    return data

class GAT(nn.Module):
    def __init__(self, channel):
        super(GAT, self).__init__()
        self.conv = BasicConv2d(channel, 64, kernel_size=1)
        self.GCN = GCNOnFeatureMap(64, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        graph_feature = construct_graph_(self.conv(feature)).cuda()
        gcn_out = self.GCN(graph_feature)
        out = gcn_out.reshape(feature.shape[0], -1, feature.shape[2], feature.shape[3])
        # print("gatout.shape = ", out.shape)
        return self.sigmoid(out)

class RRM(nn.Module):
    def __init__(self, channel, ratio=16):
        super(RRM, self).__init__()
        self.conv_cat1 = BasicConv2d(channel * 2, channel, kernel_size=1)
        self.conv3_2 = BasicConv2d(channel, channel // ratio, kernel_size=3, padding=1)
        self.conv3_3 = BasicConv2d(channel, channel // ratio, kernel_size=3, padding=1)
        self.conv_cat2 = BasicConv2d(channel // ratio * 2, channel // ratio, kernel_size=3, padding=1)
        self.conv_out = BasicConv2d(channel // ratio, channel, kernel_size=3, padding=1)
        self.ccm = CCM(channel)
        self.scm1 = SCM()
        self.gat = GAT(channel)

    def forward(self, rgb, depth, weight_d):
        feature = self.conv_cat1(torch.cat([rgb, depth], dim=1)) * weight_d
        ccm_out = self.ccm(feature) * feature
        w_scm1 = self.scm1(ccm_out)
        scm1_out1 = self.conv3_2(ccm_out * w_scm1)
        scm1_out2 = self.conv3_3(ccm_out * (1 - w_scm1))
        w_gat = self.gat(feature)
        gat_out = rgb * w_gat
        out = self.conv_out(self.conv_cat2(torch.cat([scm1_out1, scm1_out2], dim=1))) + gat_out + rgb
        return out

def bilateral_filter(input_tensor, sigma_range = 0.1, sigma_spatial = 2.0):
    # 将Pytorch Tensor转换为NuPy数组
    input_array = input_tensor.cpu().detach().numpy()
    # 初始化输出数组
    output_array = np.zeros_like(input_array)

    # 对每个样本和通道进行滤波
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            # 将特征图转换为uint8类型
            input_img = input_array[i, j].astype(np.uint8)
            # 使用OpenCV的双边滤波
            filtered_img = cv2.bilateralFilter(input_img, -1, sigma_spatial, sigma_range)
            # 将结果放回输出数组
            output_array[i, j] = filtered_img

    # 将NumPy数组转换为PyTorch Tensor并返回
    return torch.from_numpy(output_array).to(input_tensor.device)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, n_filters, 1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv3(x)
        return x