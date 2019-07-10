import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
from mmcv.runner import load_checkpoint
import logging
import math
#from pytorch_syncbn.syncbn import SyncBN # tong

# from sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback # mao

#import encoding

batch_norm = nn.BatchNorm2d # encoding.nn.SyncBatchNorm


blocks_keys = [
    'shufflenet_3x3',
    'shufflenet_5x5',
    'shufflenet_7x7',
    'xception_3x3',
]

blocks_dict = {
    'shufflenet_3x3': lambda inp, oup, mid_c, stride: Shufflenet(inp, oup, mid_c, 3, 1, stride),
    'shufflenet_5x5': lambda inp, oup, mid_c, stride: Shufflenet(inp, oup, mid_c, 5, 2, stride),
    'shufflenet_7x7': lambda inp, oup, mid_c, stride: Shufflenet(inp, oup, mid_c, 7, 3, stride),
    'xception_3x3': lambda inp, oup, mid_c, stride: Shuffle_Xception(inp, oup, mid_c, 3, 1, stride)
}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Shufflenet(nn.Module):
    def __init__(self, inp, oup, mid_c, ksize, pad, stride):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        assert mid_c == oup // 2

        mid = mid_c
        outputs = oup - inp

        if stride == 2:
            self.branch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                batch_norm(inp),#, sync=cfg.MODEL.SYNCBN_ON),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                batch_norm(inp),#, sync=cfg.MODEL.SYNCBN_ON),
                nn.ReLU(),
            )

        self.branch2 = nn.Sequential(
            # pw
            nn.Conv2d(inp, mid, 1, 1, 0, bias=False),
            batch_norm(mid),#, sync=cfg.MODEL.SYNCBN_ON),
            nn.ReLU(),
            # dw
            nn.Conv2d(mid, mid, ksize, stride, pad, groups=mid, bias=False),
            batch_norm(mid),#, sync=cfg.MODEL.SYNCBN_ON),
            # pw-linear
            nn.Conv2d(mid, outputs, 1, 1, 0, bias=False),
            batch_norm(outputs),#, sync=cfg.MODEL.SYNCBN_ON),
            nn.ReLU(),
        )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        if self.stride == 1:
            x = channel_shuffle(x, 2)
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.branch2(x2))
        elif self.stride == 2:
            out = self._concat(self.branch1(x), self.branch2(x))

        return out


class Shuffle_Xception(nn.Module):
    def __init__(self, inp, oup, mid_c, ksize, pad, stride):
        super(Shuffle_Xception, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        assert mid_c == oup // 2

        mid = mid_c
        outputs = oup - inp

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                batch_norm(inp),#, sync=cfg.MODEL.SYNCBN_ON),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                batch_norm(inp),#, sync=cfg.MODEL.SYNCBN_ON),
                nn.ReLU(),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
            batch_norm(inp),#, sync=cfg.MODEL.SYNCBN_ON),
            # pw-linear
            nn.Conv2d(inp, mid, 1, 1, 0, bias=False),
            batch_norm(mid),#, sync=cfg.MODEL.SYNCBN_ON),
            nn.ReLU(),

            nn.Conv2d(mid, mid, ksize, 1, pad, groups=mid, bias=False),
            batch_norm(mid),#, sync=cfg.MODEL.SYNCBN_ON),
            # pw-linear
            nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            batch_norm(mid),#, sync=cfg.MODEL.SYNCBN_ON),
            nn.ReLU(),

            nn.Conv2d(mid, mid, ksize, 1, pad, groups=mid, bias=False),
            batch_norm(mid),#, sync=cfg.MODEL.SYNCBN_ON),
            # pw-linear
            nn.Conv2d(mid, outputs, 1, 1, 0, bias=False),
            batch_norm(outputs),#, sync=cfg.MODEL.SYNCBN_ON),
            nn.ReLU(),
        )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.stride == 1:
            x = channel_shuffle(x, 2)
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.branch2(x2))
        elif self.stride == 2:
            out = self._concat(self.branch1(x), self.branch2(x))

        return out



