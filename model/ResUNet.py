import torch
import torch.nn as nn
import os
from torch.nn import functional as F
from model.modules import ResidualConv, ResidualConv_Firstlayer, UpsampleConv, IRC3

class ResUNet_Encoder(nn.Module):
    
    def __init__(self, 
                 in_channels, 
                 num_channels = [8, 16, 32, 64, 128],
                 conv_bias = False,
                 dropout = 0.25,
                ):
        super(ResUNet_Encoder, self).__init__()
        
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.conv_bias = conv_bias
        self.dropout = dropout

        self.conv1_ct = ResidualConv_Firstlayer(in_channels, num_channels[0], 3, 1, 1, self.dropout, self.conv_bias)
        self.conv2_ct = ResidualConv(num_channels[0], num_channels[1], 3, 2, 1, self.dropout, self.conv_bias)
        self.conv3_ct = ResidualConv(num_channels[1], num_channels[2], 3, 2, 1, self.dropout, self.conv_bias)
        self.conv4_ct = ResidualConv(num_channels[2], num_channels[3], 3, 2, 1, self.dropout, self.conv_bias)
        self.conv5_ct = ResidualConv(num_channels[3], num_channels[4], 3, 2, 1, self.dropout, self.conv_bias)
        
        self.conv1_pet = ResidualConv_Firstlayer(in_channels, num_channels[0], 3, 1, 1, self.dropout, self.conv_bias)
        self.conv2_pet = ResidualConv(num_channels[0], num_channels[1], 3, 2, 1, self.dropout, self.conv_bias)
        self.conv3_pet = ResidualConv(num_channels[1], num_channels[2], 3, 2, 1, self.dropout, self.conv_bias)
        self.conv4_pet = ResidualConv(num_channels[2], num_channels[3], 3, 2, 1, self.dropout, self.conv_bias)
        self.conv5_pet = ResidualConv(num_channels[3], num_channels[4], 3, 2, 1, self.dropout, self.conv_bias)
        
    def forward(self, ct, pet):
        enc1_ct = self.conv1_ct(ct)
        enc1_pet = self.conv1_pet(pet)
        
        enc2_ct = self.conv2_ct(enc1_ct)
        enc2_pet = self.conv2_pet(enc1_pet)
        
        enc3_ct = self.conv3_ct(enc2_ct)
        enc3_pet = self.conv3_pet(enc2_pet)
        
        enc4_ct = self.conv4_ct(enc3_ct)
        enc4_pet = self.conv4_pet(enc3_pet)
        
        enc5_ct = self.conv5_ct(enc4_ct)
        enc5_pet = self.conv5_pet(enc4_pet)
        
        return enc1_ct, enc2_ct, enc3_ct, enc4_ct, enc5_ct,\
                enc1_pet, enc2_pet, enc3_pet, enc4_pet, enc5_pet

class ResUNet_Decoder(nn.Module):
    
    def __init__(self, 
                 n_classes, 
                 num_channels = [8, 16, 32, 64, 128],
                 conv_bias = False,
                 dropout = 0.25
                ):
        super(ResUNet_Decoder, self).__init__()
        
        self.n_classes = n_classes
        self.num_channels = num_channels
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
        
        self.output5 = IRC3(num_channels[4] * 2, n_classes, self.conv_bias)
        self.output4 = IRC3(num_channels[3] * 2, n_classes, self.conv_bias)
        self.output3 = IRC3(num_channels[2] * 2, n_classes, self.conv_bias)
        self.output2 = IRC3(num_channels[1] * 2, n_classes, self.conv_bias)
        self.output1 = IRC3(num_channels[0] * 2, n_classes, self.conv_bias)
        
    
    def forward(self, enc1, enc2, enc3, enc4, enc5):
        pred5 = self.output5(enc5)
        
        out = self.upsample(enc5)
        out = torch.cat([out, enc4], dim=1)
        out = self.conv4(out)
        pred4 = self.output4(out)
        
        out = self.upsample(out)
        out = torch.cat([out, enc3], dim=1)
        out = self.conv3(out)
        pred3 = self.output3(out)
        
        out = self.upsample(out)
        out = torch.cat([out, enc2], dim=1)
        out = self.conv2(out)
        pred2 = self.output2(out)
        
        out = self.upsample(out)
        out = torch.cat([out, enc1], dim=1)
        out = self.conv1(out)
        pred1 = self.output1(out)

        return pred1, pred2, pred3, pred4, pred5
        

class ResUNet(nn.Module):
    
    def __init__(self, 
                 in_channels, 
                 n_classes, 
                 num_channels = [8, 16, 32, 64, 128]
                ):
        super(ResUNet, self).__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.num_channels = num_channels

        self.encoder = ResUNet_Encoder(in_channels, num_channels)
        self.decoder = ResUNet_Decoder(n_classes, num_channels)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
    
    def forward(self, ct, pet):
        enc1_ct, enc2_ct, enc3_ct, enc4_ct, enc5_ct,\
            enc1_pet, enc2_pet, enc3_pet, enc4_pet, enc5_pet = self.encoder(ct, pet)
        
        enc1 = torch.cat([enc1_ct, enc1_pet], dim=1)
        enc2 = torch.cat([enc2_ct, enc2_pet], dim=1)
        enc3 = torch.cat([enc3_ct, enc3_pet], dim=1)
        enc4 = torch.cat([enc4_ct, enc4_pet], dim=1)
        enc5 = torch.cat([enc5_ct, enc5_pet], dim=1)
        
        pred1, pred2, pred3, pred4, pred5 = self.decoder(enc1, enc2, enc3, enc4, enc5)
        
        pred1 = F.interpolate(pred1, size=ct.size()[-3:], mode='trilinear', align_corners=True)
        pred2 = F.interpolate(pred2, size=ct.size()[-3:], mode='trilinear', align_corners=True)
        pred3 = F.interpolate(pred3, size=ct.size()[-3:], mode='trilinear', align_corners=True)
        pred4 = F.interpolate(pred4, size=ct.size()[-3:], mode='trilinear', align_corners=True)
        pred5 = F.interpolate(pred5, size=ct.size()[-3:], mode='trilinear', align_corners=True)
        
        return pred1, pred2, pred3, pred4, pred5
    
if __name__ == "__main__":
    patch_size = (128, 128, 64)
    batch_size = 4
    num_channels = [8, 16, 32, 64, 128]
    n_classes = 1
    in_channels = 1

    device = torch.device("cuda:6")

    model = ResUNet(
                patch_size=patch_size,
                in_channels=in_channels,
                n_classes=n_classes+1,
                num_channels=num_channels,
                ).to(device)
    ct = torch.randn((batch_size, 1, *patch_size)).to(device)
    pet = torch.randn((batch_size, 1, *patch_size)).to(device)

    outputs = model(ct, pet)
    print([output.shape for output in outputs])
   
