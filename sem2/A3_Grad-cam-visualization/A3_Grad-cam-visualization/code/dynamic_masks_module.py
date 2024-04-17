import numpy as np
import torch.nn as nn
import torch
from analysis_settings import base_mask_size_

class Dynamic_MaskPair(nn.Module):
    def __init__(self, img_size=224, mask_size=224, upper_y=0, lower_y=0, upper_x=0, lower_x=0, base_mask_size=base_mask_size_, base_layer_block=1, layer_masks=np.ones([1, 1])):
        super(Dynamic_MaskPair, self).__init__()

        self.episilon = 1e-12
        self.init_weights = 0.5
        self.img_size = img_size
        self.mask_size = mask_size
        self.upper_y = upper_y
        self.lower_y = lower_y
        self.upper_x = upper_x
        self.lower_x = lower_x
        self.base_mask_size = base_mask_size
        self.base_layer_block = base_layer_block

        layer_masks = layer_masks.astype('float32')
        self.layer_masks = nn.Parameter(torch.tensor(layer_masks), requires_grad=False)
        self.mask_batch = int((self.img_size/self.mask_size) * (self.img_size/self.mask_size))

        init_mask = np.full((1, self.base_mask_size, self.base_mask_size), self.init_weights)
        init_mask = init_mask.astype('float32')
        self.mask = nn.Parameter(torch.tensor(init_mask), requires_grad=True)
        pre_mask = np.ones([self.base_mask_size, self.base_mask_size])

        for i in range(self.base_mask_size):
            for j in range(self.base_mask_size):
                pre_mask[i][j] = layer_masks[i//base_layer_block][j//base_layer_block]

        pre_mask = pre_mask.astype('float32')

        self.pre_mask = nn.Parameter(torch.tensor(pre_mask), requires_grad=False)
        self.final_mask = self.pre_mask * self.mask

    def forward(self, x):
        upsample = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear', align_corners=None)
        real_mask = upsample((self.mask * self.pre_mask).unsqueeze(0))
        x = x * real_mask
        self.final_mask = self.pre_mask * self.mask

        return x