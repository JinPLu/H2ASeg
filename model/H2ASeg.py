import torch
import torch.nn as nn
import os
from torch.nn import functional as F
from model.Attention import MCSA, Bidirectional_Spatial_Attention
from model.MCSA import ResUNet_MCSA_Encoder
from model.TAMW import ResUNet_TAMW_Decoder

class H2ASeg(nn.Module):
    
    def __init__(self, 
                 patch_size, 
                 in_channels, 
                 n_classes, 
                 num_channels = [8, 16, 32, 64, 128],
                 window_size = (8, 8, 4),
                 attention_dropout = 0.5,
                 encoder_dropout = 0.5,
                 decoder_dropout = 0.5,
                 conv_bias = True,
                ):
        super(H2ASeg, self).__init__()
        
        self.patch_size = torch.tensor(patch_size)
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.window_size = torch.tensor(window_size)
        if len(self.window_size.shape) == 1:
            self.window_size = self.window_size.repeat(5, 1)
        self.encoder = ResUNet_MCSA_Encoder(self.in_channels, self.num_channels, self.window_size, 
                                          conv_bias, encoder_dropout, attention_dropout)
        self.decoder = ResUNet_TAMW_Decoder(self.patch_size, n_classes, num_channels, conv_bias, decoder_dropout)
    
    def forward(self, ct, pet):
        encoder_outputs = self.encoder(ct, pet)
        
        pred1, pred2, pred3, pred4, pred5 = self.decoder(encoder_outputs)
        
        pred1 = F.interpolate(pred1, size=ct.size()[-3:], mode='trilinear', align_corners=True)
        pred2 = F.interpolate(pred2, size=ct.size()[-3:], mode='trilinear', align_corners=True)
        pred3 = F.interpolate(pred3, size=ct.size()[-3:], mode='trilinear', align_corners=True)
        pred4 = F.interpolate(pred4, size=ct.size()[-3:], mode='trilinear', align_corners=True)
        pred5 = F.interpolate(pred5, size=ct.size()[-3:], mode='trilinear', align_corners=True)
        
        return pred1, pred2, pred3, pred4, pred5

    
if __name__ == "__main__":
    patch_size = (128, 128, 64)
    window_size = (8, 8, 4)
    batch_size = 2
    num_channels = [8, 16, 32, 64, 128]
    n_classes = 1
    in_channels = 1

    device = torch.device("cuda:0")

    model = ResUNet_sa(
                patch_size=patch_size,
                in_channels=in_channels,
                n_classes=n_classes+1,
                num_channels=num_channels,
                window_size=window_size,
                ).to(device)
    ct = torch.randn((batch_size, 1, *patch_size)).to(device)
    pet = torch.randn((batch_size, 1, *patch_size)).to(device)

    outputs = model(ct, pet)
    print([output.shape for output in outputs])
   
