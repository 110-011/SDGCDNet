import torch
import torch.nn as nn
from toolbox.models.A__Paper.SDGCDNet.scripts import ASFE, RRM
from toolbox.models.A__Paper.SDGCDNet.scripts import bilateral_filter
from toolbox.models.A__Paper.SDGCDNet.scripts import DecoderBlock
from toolbox.backbone.mobilenetv2 import mobilenet_v2
# 仅对深度图应用双边滤波器

class fuse(nn.Module):
    def __init__(self, channel):
        super(fuse, self).__init__()
        self.bam = ASFE(channel)
        self.rrm = RRM(channel)

    def forward(self, rgb, depth):
        rgb, depth, w = self.bam(rgb, depth)
        out = self.rrm(rgb, depth, w)
        return out, rgb, depth

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        # Encoder_mobilenet_v2
        self.mobilenet_R = mobilenet_v2(pretrained=True)
        self.mobilenet_D = mobilenet_v2(pretrained=True)
        self.channels = [16, 24, 32, 160, 320]  # mobilenet_v2
        # 融合
        self.fuse_1 = fuse(self.channels[0])
        self.fuse_2 = fuse(self.channels[1])
        self.fuse_3 = fuse(self.channels[2])
        self.fuse_4 = fuse(self.channels[3])
        self.fuse_5 = fuse(self.channels[4])
        # 解码
        self.decoder_4 = DecoderBlock(self.channels[4], self.channels[3])
        self.decoder_3 = DecoderBlock(self.channels[3], self.channels[2])
        self.decoder_2 = DecoderBlock(self.channels[2], self.channels[1])
        self.decoder_1 = DecoderBlock(self.channels[1], self.channels[0])
        self.finaldeconv1 = nn.ConvTranspose2d(self.channels[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, 6, 2, padding=1)

    def forward(self, RGB, DSM):
        # RGB_Encoder_mobilenet_v2
        rgb_1 = self.mobilenet_R.features[0:2](RGB)  # [4, 16, 128, 128]
        depth_1 = self.mobilenet_D.features[0:2](DSM)  # [4, 16, 128, 1R28]
        depth_1 = bilateral_filter(depth_1) + depth_1
        f1, rgb1, depth1 = self.fuse_1(rgb_1, depth_1)

        rgb_2 = self.mobilenet_R.features[2:4](rgb_1 + f1)  # [4, 24, 64, 64]
        depth_2 = self.mobilenet_D.features[2:4](depth_1 + depth1)  # [4, 24, 64, 64]
        depth_2 = bilateral_filter(depth_2) + depth_2
        f2, rgb2, depth2 = self.fuse_2(rgb_2, depth_2)

        rgb_3 = self.mobilenet_R.features[4:7](rgb_2 + f2)  # [4, 32, 32, 32]
        depth_3 = self.mobilenet_D.features[4:7](depth_2 + depth2)  # [4, 32, 32, 32]
        depth_3 = bilateral_filter(depth_3) + depth_3
        f3, rgb3, depth3 = self.fuse_3(rgb_3, depth_3)

        rgb_4 = self.mobilenet_R.features[7:17](rgb_3 + f3)  # [4, 160, 16, 16]
        depth_4 = self.mobilenet_D.features[7:17](depth_3)  # [4, 160, 16, 16]
        depth_4 = bilateral_filter(depth_4) + depth_4
        f4, rgb4, depth4 = self.fuse_4(rgb_4, depth_4)

        rgb_5 = self.mobilenet_R.features[17:18](rgb_4 + f4)  # [4, 320, 8, 8]
        depth_5 = self.mobilenet_D.features[17:18](depth_4 + depth4)  # [4, 320, 8, 8]
        depth_5 = bilateral_filter(depth_5) + depth_5
        f5, rgb5, depth5 = self.fuse_5(rgb_5, depth_5)

        d4 = self.decoder_4(f5) + f4
        d3 = self.decoder_3(d4) + f3
        d2 = self.decoder_2(d3) + f2
        d1 = self.decoder_1(d2) + f1
        fuse = self.finaldeconv1(d1)
        fuse = self.finalrelu1(fuse)
        fuse = self.finalconv2(fuse)
        fuse = self.finalrelu2(fuse)
        fuse = self.finalconv3(fuse)
        return fuse, d1, d2, d3, d4, f1, f2, f3, f4, f5

if __name__ == '__main__':
    rgb = torch.randn(4, 3, 256, 256).cuda()
    depth = torch.randn(4, 3, 256, 256).cuda()
    net = Module().cuda()
    net.load_state_dict(torch.load('/media/user/shuju/XJ/toolbox/models/A__Paper/guass/BAMCCM_RRMCCM_DBF_S/BAMCCM_RRMCCM_DBF_S-Potsdam-loss.pth'))
    outs = net(rgb, depth)
    # print("outs.shpae = ", outs.shape)
    for out in outs:
        print("out.shape = ", out.shape)