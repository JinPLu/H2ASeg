import os
import operator

import torch
import torch.nn as nn
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

        try:
            normalized_in_channels = operator.index(in_channels)
        except TypeError as exc:
            raise ValueError(
                "in_channels must be the integer 1 for the single-channel CT/PET architecture"
            ) from exc
        if isinstance(in_channels, bool) or normalized_in_channels != 1:
            raise ValueError(
                "in_channels must be the integer 1 for the single-channel CT/PET architecture"
            )
        try:
            normalized_n_classes = operator.index(n_classes)
        except TypeError as exc:
            raise ValueError(
                "n_classes must be the integer 2 for the binary H2ASeg architecture"
            ) from exc
        if isinstance(n_classes, bool) or normalized_n_classes != 2:
            raise ValueError(
                "n_classes must be the integer 2 for the binary H2ASeg architecture"
            )
        
        self.patch_size = torch.tensor(patch_size)
        self.in_channels = normalized_in_channels
        self.n_classes = normalized_n_classes
        self.num_channels = num_channels
        self.window_size = torch.tensor(window_size)
        if len(self.window_size.shape) == 1:
            self.window_size = self.window_size.repeat(5, 1)
        self.encoder = ResUNet_MCSA_Encoder(self.in_channels, self.num_channels, self.window_size, 
                                          conv_bias, encoder_dropout, attention_dropout)
        self.decoder = ResUNet_TAMW_Decoder(
            self.patch_size,
            self.n_classes,
            num_channels,
            conv_bias,
            decoder_dropout,
        )
    
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
    patch_size = (32, 32, 16)
    window_size = (4, 4, 2)
    batch_size = 1
    num_channels = [4, 4, 4, 8, 16]
    n_classes = 2
    in_channels = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = H2ASeg(
                patch_size=patch_size,
                in_channels=in_channels,
                n_classes=n_classes,
                num_channels=num_channels,
                window_size=window_size,
                ).eval().to(device)
    ct = torch.randn((batch_size, 1, *patch_size)).to(device)
    pet = torch.randn((batch_size, 1, *patch_size)).to(device)

    with torch.inference_mode():
        outputs = model(ct, pet)
    print([output.shape for output in outputs])
   
