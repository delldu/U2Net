"""matting appliction class."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 02月 27日 星期六 17:25:55 CST
# ***
# ************************************************************************************/
#

import inspect
import os

import torch.nn as nn

from .model import get_model


class matting(nn.Module):
    """matting."""

    def __init__(self, weight_file="matting.pth"):
        """Init model."""
        super(matting, self).__init__()
	dir = os.path.dirname(inspect.getfile(self.__init__))
        checkpoint = os.path.join(dir, '/models/%s' % (weight_file))
        model, device = get_model(checkpoint)

        self.model = model
        self.device = device

    def forward(self, x):
        """Forward."""

        return self.model(x)
