import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
from mobilenetv3.mobilenetv3 import Block,SeModule,OpticalInceptionBlock
from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from torch.nn import init


class Backbone_MOBILE_EncoderFirstStage(nn.Module):
    """
    The final FPN version of MobileNetV3-Large, designed to mimic the
    downsampling rhythm and depth of ResNet-IR-SE50 for E2Style-like encoders.
    """
    def __init__(self, act=nn.Hardswish):
        super(Backbone_MOBILE_EncoderFirstStage, self).__init__()
        
        # MODIFICATION 1: Initial conv layer now has stride=1 (NO initial downsampling)
        self.input_layer = OpticalInceptionBlock(in_channels=1, out_channels_per_branch=4, device=torch.device("cuda"))   #TODO:channel_in
       # self.input_layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)
        self.se = SeModule(in_size=16)
        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(256 * 7 * 7, 512 * 9))
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        # Using the FULL 15 blocks from MobileNetV3-Large, with modified strides
        # to delay downsampling and increase depth at each stage.
        self.bneck = nn.Sequential(
            # --- Stage 0: Processing at H x W (Full) resolution ---
            Block(3, 16, 16, 16, nn.ReLU, True, 1),   # bneck[0]
            
            # --- Stage 1: Downsample to H/2 and process ---
            Block(3, 16, 64, 24, nn.ReLU, True, 1),   # bneck[1]
            Block(3, 24, 72, 64, nn.ReLU, True, 2),   # bneck[2]   downsampling to H/2
            
            # --- Stage 2: Downsample to H/4 and process ---
            # MODIFICATION 2: Delayed downsampling from block 3 to block 6
            Block(5, 64, 72, 40, nn.ReLU, True, 1),    # bneck[3]: 
            Block(5, 40, 120, 40, nn.ReLU, True, 1),   # bneck[4]
            Block(5, 40, 120, 40, nn.ReLU, True, 1),   # bneck[5]
            Block(3, 40, 240, 128, act, True, 2),      # bneck[6]:  downsampling to H/4
            Block(3, 128, 200, 80, act, True, 1),      # bneck[7] 
            Block(3, 80, 184, 80, act, True, 1),      # bneck[8]
            Block(3, 80, 184, 80, act, True, 1),      # bneck[9]
            
            # --- Stage 3: Downsample to H/8 and process ---
            # MODIFICATION 3: Kept subsequent blocks for depth
            Block(3, 80, 480, 112, act, True, 1),      # bneck[10]
            Block(3, 112, 672, 112, act, True, 1),     # bneck[11]
            Block(5, 112, 672, 160, act, True, 1),     # bneck[12]
            Block(5, 160, 672, 160, act, True, 1),     # bneck[13]
            Block(5, 160, 960, 256, act, True, 2),     # bneck[14]  downsampling to H/8
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 1. Process at H x W resolution
        features = []
        x = self.se(self.hs1(self.bn1(self.input_layer(x))))
        for l in self.bneck[:3]:  
            x = l(x)
        features.append(x)    
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512) 
        
        for l in self.bneck[3:7]:
            x = l(x)
        features.append(x)     
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)  
            
        for l in self.bneck[7:15]:
            x = l(x)
        features.append(x)     
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)    

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        
        return x
        #return {'latents': x,'features':features}


class BackboneEncoderFirstStage(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderFirstStage, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(256 * 7 * 7, 512 * 9))
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)
    def forward(self, x):
        features = []
        x = self.input_layer(x)
        for l in self.modulelist[:3]:
          x = l(x)     #(4,64,128,128)      C:64  H/2,W/2  
        features.append(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512)   #torch.Size([4, 4, 512])   LATENT CODE:1-4 coarse
        for l in self.modulelist[3:7]:
          x = l(x)  #新增     #torch.Size([4, 128, 64, 64])    C:128 H/4.W/4      
        features.append(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)    # Torch.Size([4, 5, 512])  LATENT CODE:5-9 MID   
        for l in self.modulelist[7:21]:
          x = l(x)     #(4,256,32,32)     C:256,H/8,W/8  
        features.append(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)  #(4,9,512)    LATENT CODE:10-18 DETAIL 
        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        
        return {'latents': x,'features':features}
class BackboneEncoderRefineStage(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderRefineStage, self).__init__()
        # print('Using BackboneEncoderRefineStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(6, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(256 * 7 * 7, 512 * 9))
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    def forward(self, x, first_stage_output_image):
        features = []
        x = torch.cat((x,first_stage_output_image), dim=1)
        x = self.input_layer(x)
        
        for l in self.modulelist[:3]:
          x = l(x)
        features.append(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512)
        
        for l in self.modulelist[3:7]:
          x = l(x)
        features.append(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)
        
        for l in self.modulelist[7:21]:
          x = l(x)
        features.append(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        
        return {'latents': x,'features':features}