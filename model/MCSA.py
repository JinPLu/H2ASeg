import torch
import torch.nn as nn
import os
from torch.nn import functional as F
from model.Attention import MCSA, Bidirectional_Spatial_Attention
from model.ResUNet import ResUNet_Encoder, ResUNet_Decoder

class ResUNet_MCSA_Encoder(ResUNet_Encoder):
    def __init__(self, 
                 in_channels, 
                 num_channels = [8, 16, 32, 64, 128], 
                 window_size = (8, 8, 4),
                 conv_bias = False,
                 dropout = 0.3,
                 att_dropout = 0.25,):
        super(ResUNet_MCSA_Encoder, self).__init__(in_channels, num_channels, conv_bias, dropout)
        self.sa2 = MCSA(num_channels[1], window_size[1], att_dropout)
        self.sa3 = MCSA(num_channels[2], window_size[2], att_dropout)
        
        self.sa4 = Bidirectional_Spatial_Attention(num_channels[3], num_channels[3], att_dropout, skip_connection=True)
        self.sa5 = Bidirectional_Spatial_Attention(num_channels[4], num_channels[4], att_dropout, skip_connection=True)
        
    def forward(self, ct, pet):
        # Encode
        enc1_ct = self.conv1_ct(ct)
        enc1_pet = self.conv1_pet(pet)
        
        enc2_ct = self.conv2_ct(enc1_ct)
        enc2_pet = self.conv2_pet(enc1_pet)
        enc2_ct, enc2_pet = self.sa2(enc2_ct, enc2_pet)
        
        enc3_ct = self.conv3_ct(enc2_ct)
        enc3_pet = self.conv3_pet(enc2_pet)
        enc3_ct, enc3_pet = self.sa3(enc3_ct, enc3_pet)
        
        enc4_ct = self.conv4_ct(enc3_ct)
        enc4_pet = self.conv4_pet(enc3_pet)
        enc4_ct, enc4_pet = self.sa4(enc4_ct, enc4_pet)
        
        enc5_ct = self.conv5_ct(enc4_ct)
        enc5_pet = self.conv5_pet(enc4_pet)
        enc5_ct, enc5_pet = self.sa5(enc5_ct, enc5_pet)
        
        return enc1_ct, enc2_ct, enc3_ct, enc4_ct, enc5_ct,\
                enc1_pet, enc2_pet, enc3_pet, enc4_pet, enc5_pet
        
