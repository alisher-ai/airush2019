import torch
import torchvision.models as models
import torch.nn as nn
from torchsummary import summary

import torch,math,sys
import torch.utils.model_zoo as model_zoo
from functools import partial
# from ...torch_core import Module


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Baseline(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.Conv2d(32, 128, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.5),
            nn.Conv2d(1024, out_size, 4, 1),
        )

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class Resnet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.resnet18(pretrained=True)
        # model = models.densenet121(pretrained=True)
        # print(model)
        # num_ftrs = model.classifier.in_features
        # model.classifier = nn.Linear(num_ftrs, out_size)
        # print(model)
        model = list(model.children())[:-1]
        # print(model)
        # model.append(nn.Conv2d(512, out_size, 1))
        model.append(Flatten())
        model.append(nn.Linear(512, 1024))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Dropout(0.5))
        # model.append(nn.Linear(1024, 1024))
        # model.append(nn.ReLU(inplace=True))
        model.append(nn.Linear(1024, out_size))
        self.net = nn.Sequential(*model)
        # self.net = model

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


#
# act_fn = nn.ReLU(inplace=True)
#
#
# def init_cnn(m):
#     if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
#     if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
#     for l in m.children(): init_cnn(l)
#
# def conv(ni, nf, ks=3, stride=1, bias=False):
#     return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)
#
# def noop(x): return x
#
# def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
#     bn = nn.BatchNorm2d(nf)
#     nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
#     layers = [conv(ni, nf, ks, stride=stride), bn]
#     if act: layers.append(act_fn)
#     return nn.Sequential(*layers)
#
# class ResBlock(Module):
#     def __init__(self, expansion, ni, nh, stride=1):
#         nf,ni = nh*expansion,ni*expansion
#         layers  = [conv_layer(ni, nh, 3, stride=stride),
#                    conv_layer(nh, nf, 3, zero_bn=True, act=False)
#         ] if expansion == 1 else [
#                    conv_layer(ni, nh, 1),
#                    conv_layer(nh, nh, 3, stride=stride),
#                    conv_layer(nh, nf, 1, zero_bn=True, act=False)
#         ]
#         self.convs = nn.Sequential(*layers)
#         # TODO: check whether act=True works better
#         self.idconv = noop if ni==nf else conv_layer(ni, nf, 1, act=False)
#         self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)
#
#     def forward(self, x): return act_fn(self.convs(x) + self.idconv(self.pool(x)))
#
# def filt_sz(recep): return min(64, 2**math.floor(math.log2(recep*0.75)))
#
# class XResNet(nn.Sequential):
#     def __init__(self, expansion, layers, c_in=3, c_out=1000):
#         stem = []
#         sizes = [c_in,32,32,64]
#         for i in range(3):
#             stem.append(conv_layer(sizes[i], sizes[i+1], stride=2 if i==0 else 1))
#             #nf = filt_sz(c_in*9)
#             #stem.append(conv_layer(c_in, nf, stride=2 if i==1 else 1))
#             #c_in = nf
#
#         block_szs = [64//expansion,64,128,256,512]
#         blocks = [self._make_layer(expansion, block_szs[i], block_szs[i+1], l, 1 if i==0 else 2)
#                   for i,l in enumerate(layers)]
#         super().__init__(
#             *stem,
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             *blocks,
#             nn.AdaptiveAvgPool2d(1), Flatten(),
#             nn.Linear(block_szs[-1]*expansion, c_out),
#         )
#         init_cnn(self)
#
#     def _make_layer(self, expansion, ni, nf, blocks, stride):
#         return nn.Sequential(
#             *[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1)
#               for i in range(blocks)])
#
# def xresnet(expansion, n_layers, name, pretrained=False, **kwargs):
#     model = XResNet(expansion, n_layers, **kwargs)
#     if pretrained: model.load_state_dict(model_zoo.load_url(model_urls[name]))
#     return model
#
# me = sys.modules[__name__]
# for n,e,l in [
#     [ 18 , 1, [2,2,2 ,2] ],
#     [ 34 , 1, [3,4,6 ,3] ],
#     [ 50 , 4, [3,4,6 ,3] ],
#     [ 101, 4, [3,4,23,3] ],
#     [ 152, 4, [3,8,36,3] ],
# ]:
#     name = f'xresnet{n}'
#     setattr(me, name, partial(xresnet, expansion=e, n_layers=l, name=name))


