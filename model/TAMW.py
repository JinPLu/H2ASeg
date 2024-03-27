import torch
import torch.nn as nn
from torch.nn import functional as F
from model.modules import IRC3, IRC1, ResidualConv, UpsampleConv
from model.Attention import Adaptive_Weighting
from model.ResUNet import ResUNet_Encoder

def concat(*args):
    return torch.cat(args, dim=1)

class TAMW(nn.Module):
    def __init__(self, 
                 in_channels1,
                 in_channels2,
                 input_size,
                 n_classes,
                 conv_bias=False):
        super(TAMW, self).__init__()
        
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.input_size = input_size
        self.n_classes = n_classes
        self.conv_bias = conv_bias
        self.softmax = nn.Softmax(1)
        self.upscale = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.CA_fore = Adaptive_Weighting(self.in_channels1, self.in_channels2, self.input_size)
        self.CA_back = Adaptive_Weighting(self.in_channels1, self.in_channels2, self.input_size)

    def forward(self, ct, pet, f_hi, p_hi):
        '''
        ct/pet: ct/pet feature of current-level encoder
        f_hi: feature of high-level decoder
        p_hi: prediction map of high-level decoder 
        '''
        # ct_pet: (b, 2c, h, w, d)
        # f_hi: (b, 2c, h, w, d)
        p_hi = self.softmax(self.upscale(p_hi))
        ct_fore = ct * p_hi[:, 1:2]
        ct_back = ct * p_hi[:, 0:1]
        pet_fore = pet * p_hi[:, 1:2]
        pet_back = pet * p_hi[:, 0:1]
        
        f_fore = self.CA_fore(ct_fore, pet_fore, f_hi)
        f_back = self.CA_back(ct_back, pet_back, f_hi)

        f_enhance = concat(ct, pet, f_hi) + concat(*f_fore) - concat(*f_back)
        return f_enhance


class ResUNet_TAMW_Decoder(nn.Module):
    
    def __init__(self,
                 patch_size,
                 n_classes,
                 num_channels = [8, 16, 32, 64, 128],
                 conv_bias = False,
                 dropout = 0.5):
        super(ResUNet_TAMW_Decoder, self).__init__()
        
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.conv_bias = conv_bias
        self.dropout = dropout
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        
        self.conv4 = ResidualConv(num_channels[3] * 2 + num_channels[4] * 2, 
                                  num_channels[3] * 2, 3, 1, 1, self.dropout, self.conv_bias)
        self.conv3 = ResidualConv(num_channels[2] * 2 + num_channels[3] * 2, 
                                  num_channels[2] * 2, 3, 1, 1, self.dropout, self.conv_bias)
        self.conv2 = ResidualConv(num_channels[1] * 2 + num_channels[2] * 2, 
                                  num_channels[1] * 2, 3, 1, 1, self.dropout, self.conv_bias)
        self.conv1 = ResidualConv(num_channels[0] * 2 + num_channels[1] * 2, 
                                  num_channels[0] * 2, 3, 1, 1, self.dropout, self.conv_bias)
        
        self.tamw4 = TAMW(num_channels[3], num_channels[4], self.patch_size // 8, self.n_classes, self.conv_bias)
        self.tamw3 = TAMW(num_channels[2], num_channels[3], self.patch_size // 4, self.n_classes, self.conv_bias)
        self.tamw2 = TAMW(num_channels[1], num_channels[2], self.patch_size // 2, self.n_classes, self.conv_bias)
        self.tamw1 = TAMW(num_channels[0], num_channels[1], self.patch_size     , self.n_classes, self.conv_bias)
        
        self.output5 = IRC3(num_channels[4] * 2, n_classes, self.conv_bias)
        self.output4 = IRC3(num_channels[3] * 2, n_classes, self.conv_bias)
        self.output3 = IRC3(num_channels[2] * 2, n_classes, self.conv_bias)
        self.output2 = IRC3(num_channels[1] * 2, n_classes, self.conv_bias)
        self.output1 = IRC3(num_channels[0] * 2, n_classes, self.conv_bias)
    
    
    def forward(self, encoder_outputs):
        context_ct1, context_ct2, context_ct3, context_ct4, out_ct, \
            context_pet1, context_pet2, context_pet3, context_pet4, out_pet = encoder_outputs
        
        out = concat(out_ct, out_pet)
        pred5 = self.output5(out)
        
        out = self.upsample(out)
        out = self.tamw4(context_ct4, context_pet4, out, pred5)
        out = self.conv4(out)
        pred4 = self.output4(out)
        
        out = self.upsample(out)
        out = self.tamw3(context_ct3, context_pet3, out, pred4)
        out = self.conv3(out)
        pred3 = self.output3(out)
        
        out = self.upsample(out)
        out = self.tamw2(context_ct2, context_pet2, out, pred3)
        out = self.conv2(out)
        pred2 = self.output2(out)
        
        out = self.upsample(out)
        out = self.tamw1(context_ct1, context_pet1, out, pred2)
        out = self.conv1(out)
        pred1 = self.output1(out)
        
        return pred1, pred2, pred3, pred4, pred5

