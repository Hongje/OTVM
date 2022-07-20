from numpy import not_equal
import torch
import torch.nn as nn
from . import resnet_GN_WS
from . import layers_WS as L
from . import resnet_bn

FEAT_DIM = 2048
DEC_DIM = 256

def FBA(refinement):
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch='resnet50_GN_WS')
    net_decoder = builder.build_decoder(arch="fba_decoder", batch_norm=False)

    model = MattingModule(net_encoder, net_decoder, refinement)

    return model


class MattingModule(nn.Module):
    def __init__(self, net_enc, net_dec, refinement):
        super(MattingModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.refinement = refinement
        if refinement:
            self.refine = RefinementModule()
        else:
            self.refine = None

    def forward(self, x, extras):
        image, two_chan_trimap = extras
        conv_out, indices = self.encoder(x)

        hid, output, x_dec = self.decoder(conv_out, image, indices, two_chan_trimap)
        pred_alpha = output[:, :1]
        
        if self.refine is not None:
            hid, refine_output, refine_trimap = self.refine(x_dec, image, two_chan_trimap, pred_alpha)
        else:
            refine_output = None
            refine_trimap = None

        return output, hid, refine_output, refine_trimap


class ModelBuilder():
    def build_encoder(self, arch='resnet50_GN', num_channels_additional=None):
        if arch == 'resnet50_GN_WS':
            orig_resnet = resnet_GN_WS.__dict__['l_resnet50']()
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8, num_channels_additional=num_channels_additional)
        elif arch == 'resnet50_BN':
            orig_resnet = resnet_bn.__dict__['l_resnet50']()
            net_encoder = ResnetDilatedBN(orig_resnet, dilate_scale=8, num_channels_additional=num_channels_additional)
        elif arch == 'resnet18_GN_WS':
            orig_resnet = resnet_GN_WS.__dict__['l_resnet18']()
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8, num_channels_additional=num_channels_additional)
        elif arch == 'resnet34_GN_WS':
            orig_resnet = resnet_GN_WS.__dict__['l_resnet34']()
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8, num_channels_additional=num_channels_additional)

        else:
            raise Exception('Architecture undefined!')

        num_channels = 3 + 6 + 2

        if(num_channels > 3):
            print(f'modifying input layer to accept {num_channels} channels')
            net_encoder_sd = net_encoder.state_dict()
            conv1_weights = net_encoder_sd['conv1.weight']

            c_out, c_in, h, w = conv1_weights.size()
            conv1_mod = torch.zeros(c_out, num_channels, h, w)
            conv1_mod[:, :3, :, :] = conv1_weights

            conv1 = net_encoder.conv1
            conv1.in_channels = num_channels
            conv1.weight = torch.nn.Parameter(conv1_mod)

            net_encoder.conv1 = conv1

            net_encoder_sd['conv1.weight'] = conv1_mod

            net_encoder.load_state_dict(net_encoder_sd)
        return net_encoder

    def build_decoder(self, arch='fba_decoder', batch_norm=False, memory_decoder=False):
        if arch == 'fba_decoder':
            net_decoder = fba_decoder(batch_norm=batch_norm, memory_decoder=memory_decoder)

        return net_decoder


class ResnetDilatedBN(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, num_channels_additional=None):
        super(ResnetDilatedBN, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.num_channels_additional = num_channels_additional
        if self.num_channels_additional is not None:
            self.conv1_a = resnet_bn.conv3x3(self.num_channels_additional, 64, stride=2)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = [x]
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out, indices
        return [x]


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, num_channels_additional=None):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.num_channels_additional = num_channels_additional
        if self.num_channels_additional is not None:
            self.conv1_a = resnet_GN_WS.L.Conv2d(self.num_channels_additional, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, x_a=None):
        conv_out = [x]                          # OS=1
        if self.num_channels_additional is None:
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.conv1(x) + self.conv1_a(x_a)
            x = self.relu(self.bn1(x))
        conv_out.append(x)                      # OS=2
        x, indices = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)                      # OS=4
        x = self.layer2(x)
        conv_out.append(x)                      # OS=8
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        return conv_out, indices


def norm(dim, bn=False):
    if(bn is False):
        return nn.GroupNorm(32, dim)
    else:
        return nn.BatchNorm2d(dim)


def fba_fusion(alpha, img, F, B):
    F = ((alpha * img + (1 - alpha**2) * F - alpha * (1 - alpha) * B))
    B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

    F = torch.clamp(F, 0, 1)
    B = torch.clamp(B, 0, 1)
    la = 0.1
    alpha = (alpha * la + torch.sum((img - B) * (F - B), 1, keepdim=True)) / (torch.sum((F - B) * (F - B), 1, keepdim=True) + la)
    alpha = torch.clamp(alpha, 0, 1)
    return alpha, F, B


class fba_decoder(nn.Module):
    def __init__(self, batch_norm=False, memory_decoder=False):
        super(fba_decoder, self).__init__()
        pool_scales = (1, 2, 3, 6)
        self.batch_norm = batch_norm
        self.memory_decoder = memory_decoder

        self.ppm = []

        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                L.Conv2d(FEAT_DIM, DEC_DIM, kernel_size=1, bias=True),
                norm(DEC_DIM, self.batch_norm),
                nn.LeakyReLU()
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_up1 = nn.Sequential(
            L.Conv2d(FEAT_DIM + len(pool_scales) * DEC_DIM, DEC_DIM,
                     kernel_size=3, padding=1, bias=True),

            norm(DEC_DIM, self.batch_norm),
            nn.LeakyReLU(),
            L.Conv2d(DEC_DIM, DEC_DIM, kernel_size=3, padding=1),
            norm(DEC_DIM, self.batch_norm),
            nn.LeakyReLU()
        )

        # if not self.memory_decoder:
        self.conv_up2 = nn.Sequential(
            L.Conv2d((FEAT_DIM//8) + DEC_DIM, DEC_DIM,
                    kernel_size=3, padding=1, bias=True),
            norm(DEC_DIM, self.batch_norm),
            nn.LeakyReLU()
        )
        if(self.batch_norm):
            d_up3 = 128
        else:
            d_up3 = 64
        self.conv_up3 = nn.Sequential(
            L.Conv2d(DEC_DIM + d_up3, 64,
                    kernel_size=3, padding=1, bias=True),
            norm(64, self.batch_norm),
            nn.LeakyReLU()
        )

        self.unpool = nn.MaxUnpool2d(2, stride=2)

        self.conv_up4 = nn.Sequential(
            nn.Conv2d(64 + 3 + 3 + 2, 32,
                    kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16,
                    kernel_size=3, padding=1, bias=True),

            nn.LeakyReLU(),
            nn.Conv2d(16, 7, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, conv_out, img, indices, two_chan_trimap, extract_feature=False, x=None):
        # if extract_feature:
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_up1(ppm_out)
        #     return x
        # else:
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-4]), 1)

        x = self.conv_up2(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-5]), 1)
        x = self.conv_up3(x)

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, conv_out[-6][:, :3], img), 1)
        x2 = torch.cat((x, two_chan_trimap), 1)

        hid = self.conv_up4[:-1](x2)
        output = self.conv_up4[-1:](hid)

        alpha = torch.clamp(output[:, 0][:, None], 0, 1)
        F = torch.sigmoid(output[:, 1:4])
        B = torch.sigmoid(output[:, 4:7])

        # FBA Fusion
        alpha, F, B = fba_fusion(alpha, img, F, B)

        output = torch.cat((alpha, F, B), 1)

        return hid, output, x


class RefinementModule(nn.Module):
    def __init__(self, batch_norm=False):
        super(RefinementModule, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Sequential(
            L.Conv2d((64 + 3 + 3) + 2 + 1, 64,
                    kernel_size=3, padding=1, bias=True),
            norm(64, self.batch_norm),
            nn.LeakyReLU()
        )
        self.layer1 = resnet_GN_WS.BasicBlock(64, 64)
        self.layer2 = resnet_GN_WS.BasicBlock(64, 64)
        outdim = 10
        self.pred = nn.Sequential(
            nn.Conv2d(64, 32,
                      kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16,
                      kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, outdim, kernel_size=1, padding=0, bias=True)
        )
    def forward(self, x, img, two_chan_trimap, pred_alpha):
        x = torch.cat((x, two_chan_trimap, pred_alpha), 1)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pred[:-1](x)
        output = self.pred[-1](x)

        a = output[:, :7]
        alpha = torch.clamp(a[:, 0][:, None], 0, 1)
        F = torch.sigmoid(a[:, 1:4])
        B = torch.sigmoid(a[:, 4:7])
        # FBA Fusion
        alpha, F, B = fba_fusion(alpha, img, F, B)
        alpha = torch.cat((alpha, F, B), 1)
        
        trimap = output[:, -3:]

        return x, alpha, trimap
