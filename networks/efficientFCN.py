import torchvision.models as model
from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F

class HGDModule(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(HGDModule, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels * 3, center_channels, 1, bias=False),
            # norm_layer(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels))
        self.norm_center = nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0 = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1 = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_up = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

        # for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, norm_layer):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

    def forward(self, x, guide1, guide2):
        n, c, h, w = x.size()
        n1, c1, h1, w1 = guide1.size()
        n2, c2, h2, w2 = guide2.size()
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)
        x_cat = torch.cat([guide2_down, guide1_down, x], 1)
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)
        f_cat = self.conv_cat(x_cat)
        f_center = self.conv_center(x_cat)
        f_cat = f_cat.view(n, self.out_channels, h * w)
        # f_x = x_cat.view(n, 2*c, h*w)
        f_center_norm = f_center.view(n, self.center_channels, h * w)
        f_center_norm = self.norm_center(f_center_norm)
        # n x * in_channels x center_channels
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))

        ########################################
        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)

        ###################################
        # f_affinity = self.conv_affinity(guide_cat)
        guide_cat_conv = self.conv_affinity0(guide_cat)
        guide_cat_value_avg = guide_cat_conv + value_avg
        f_affinity = self.conv_affinity1(guide_cat_value_avg)
        n_aff, c_ff, h_aff, w_aff = f_affinity.size()
        f_affinity = f_affinity.view(n_aff, c_ff, h_aff * w_aff)
        norm_aff = ((self.center_channels) ** -.5)
        # x_up = norm_aff * x_center.bmm(f_affinity.transpose(1, 2))
        x_up = norm_aff * x_center.bmm(f_affinity)
        x_up = x_up.view(n, self.out_channels, h_aff, w_aff)
        x_up_cat = torch.cat([x_up, guide_cat_conv], 1)
        x_up_conv = self.conv_up(x_up_cat)
        outputs = [x_up_conv]
        return tuple(outputs)


class HGDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=nn.BatchNorm2d):
        super(HGDecoder, self).__init__()
        self.conv50 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1, padding=0, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, padding=0, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True))

        self.num_center = num_center
        self.hgdmodule0 = HGDModule(512, self.num_center, 1024, norm_layer=norm_layer)
        self.conv_pred3 = nn.Sequential(nn.Dropout2d(0.1, False),
                                        nn.Conv2d(1024, out_channels, 1, padding=0))


    def forward(self, *inputs):
        feat50 = self.conv50(inputs[-1])
        feat40 = self.conv40(inputs[-2])
        # feat30 = self.conv30(inputs[-3])
        outs0 = list(self.hgdmodule0(feat50, feat40, inputs[-3]))
        outs_pred3 = self.conv_pred3(outs0[0])
        outs = [outs_pred3]

        return tuple(outs)

class efficientFCN(nn.Module):
    def __init__(self):
        super(efficientFCN, self).__init__()

        self.pretrained = model.resnet101(pretrained=True)
        self.head = HGDecoder(2048, out_channels=1, num_center=256)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # get 4 layers
        imsize = x.size()[2:]

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        # c1 torch.Size([1, 256, 128, 128])
        # c2 torch.Size([1, 512, 64, 64])
        # c3 torch.Size([1, 1024, 32, 32])
        # c4 orch.Size([1, 2048, 16, 16])
        features = c1, c2, c3, c4

        x = list(self.head(*features))
        res = F.interpolate(x[0], imsize)
        res = self.sigmoid(res)

        return res

def get_efficientFCN_model():
    model = efficientFCN()
    return model

if __name__ == '__main__':
    device = torch.device('cuda')  # cuda:0
    inputs = torch.rand(3, 512, 512).unsqueeze(0).to(device)
    print(inputs.shape)

    net = get_efficientFCN_model().to(device)
    res = net(inputs)  # res是一个tuple类型
    print('res shape:', res.shape)

    # from thop import profile, clever_format

    # flops, params = profile(net, inputs=(inputs,))
    # macs, params = clever_format([flops, params], "%.3f")
    # print('flops:', macs)
    # print('params:', params)

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print("The number of parameters : %.3f M" % (num_params/1e6))