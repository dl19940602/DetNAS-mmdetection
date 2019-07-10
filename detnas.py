import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..registry import BACKBONES
from torch.nn import init
from mmcv.runner import load_checkpoint
import math
import numpy as np
from .shufflenet_block import *
import logging


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        batch_norm(oup),#, sync=use_sync),
        nn.ReLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        batch_norm(oup),#, sync=use_sync),
        nn.ReLU()
    )


class Select_one_OP(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Select_one_OP, self).__init__()
        self._ops = nn.ModuleList()
        for key in blocks_keys:
            op = blocks_dict[key](inp, oup, oup // 2, stride)
            self._ops.append(op)

    def forward(self, x, id):
        return self._ops[id](x)

@BACKBONES.register_module
class DetNAs(nn.Module):
    def __init__(self, det=None, norm_eval=True):
        super(DetNAs, self).__init__()
        self.norm_eval = norm_eval
        self.stage_repeats = [8, 8, 16, 8] # [4, 4, 8, 4]
        self.stage_ends_idx = [8-1, 16-1, 32-1, 40-1] #[4-1, 8-1, 16-1, 20-1]

        self.num_states = np.sum(self.stage_repeats)

        #self.stage_out_channels = [-1, 48, 96, 240, 480, 960, 1024] # [-1, 16, 64, 160, 320, 640, 1024]
#        self.stage_out_channels = [-1, 48, 96, 256, 512, 1024, 2048]
        self.stage_out_channels = [-1, 128, 256, 512, 1024, 2048]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(Select_one_OP(input_channel, output_channel, 2))
                else:
                    self.features.append(Select_one_OP(input_channel // 2, output_channel, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # TODO: whether to use this
        # self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    # def _freeze_backbone(self, freeze_at):
    #     for stage_index in range(freeze_at):
    #         if stage_index == 0:
    #             m = self.stem  # stage 0 is the stem
    #         else:
    #             m = getattr(self, "layer" + str(stage_index))
    #         for p in m.parameters():
    #             p.requires_grad = False
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        outputs = []
        rngs = np.random.randint(low=0, high=4, size=(40))
        x = self.conv1(x)
        for i, select_op in enumerate(self.features):
            x = select_op(x, rngs[i])
            if i in self.stage_ends_idx:
                outputs.append(x)
        return outputs

    def train(self, mode=True):
        super(DetNAs, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
            # trick: eval have effect on BatchNorm only
                if isinstance(m, (nn.BatchNorm2d)):
                    m.eval()


