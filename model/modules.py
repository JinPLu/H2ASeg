from torch import nn


class IRC3(nn.Module):
    
    def __init__(self, 
                 feat_in, 
                 feat_out,
                 conv_bias=False):
        super(IRC3, self).__init__()
        
        self.conv = nn.Sequential(
                nn.InstanceNorm3d(feat_out),
                nn.PReLU(),
                nn.Conv3d(feat_in, feat_out, 3, 1, 1, bias=conv_bias),
                )
    def forward(self, x):
        return self.conv(x)
    
class IRC1(nn.Module):
    
    def __init__(self, 
                 feat_in, 
                 feat_out,
                 conv_bias=False):
        super(IRC1, self).__init__()
        
        self.conv = nn.Sequential(
                nn.InstanceNorm3d(feat_out),
                nn.PReLU(),
                nn.Conv3d(feat_in, feat_out, 1, 1, 0, bias=conv_bias),
                )
    def forward(self, x):
        return self.conv(x)
    
    
class RC3(nn.Module):
    
    def __init__(self, 
                 feat_in, 
                 feat_out,
                 conv_bias=False):
        super(RC3, self).__init__()
        
        self.conv = nn.Sequential(
                nn.PReLU(),
                nn.Conv3d(feat_in, feat_out, 3, 1, 1, bias=conv_bias),
                )
    def forward(self, x):
        return self.conv(x)
    
class RC1(nn.Module):
    
    def __init__(self, 
                 feat_in, 
                 feat_out,
                 conv_bias=False):
        super(RC1, self).__init__()
        
        self.conv = nn.Sequential(
                nn.PReLU(),
                nn.Conv3d(feat_in, feat_out, 1, 1, 0, bias=conv_bias),
                )
    def forward(self, x):
        return self.conv(x)

class ResidualConv_Firstlayer(nn.Module):
    
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 kernel_size = 3,
                 stride = 1, 
                 padding = 1,
                 dropout = 0.25,
                 conv_bias = True):
        super(ResidualConv_Firstlayer, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size, stride, padding, bias=conv_bias),
            nn.InstanceNorm3d(output_dim),
            nn.PReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(output_dim, output_dim, 3, 1, 1, bias=conv_bias),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size, stride, padding, bias=conv_bias)
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class ResidualConv(nn.Module):
    
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 kernel_size = 3,
                 stride = 2, 
                 padding = 1,
                 dropout = 0.25,
                 conv_bias = False):
        super(ResidualConv, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.InstanceNorm3d(input_dim),
            nn.PReLU(),
            nn.Conv3d(input_dim, output_dim, kernel_size, stride, padding, bias=conv_bias),
            nn.Dropout3d(dropout),
            nn.InstanceNorm3d(output_dim),
            nn.PReLU(),
            nn.Conv3d(output_dim, output_dim, 3, 1, 1, bias=conv_bias),
        )
        self.conv_skip = nn.Sequential(
            nn.InstanceNorm3d(input_dim),
            nn.PReLU(),
            nn.Conv3d(input_dim, output_dim, 3, stride, 1, bias=conv_bias),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)
    
    
class Upsample(nn.Module):
    
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 kernel,
                 stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose3d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)
    
class UpsampleConv(nn.Module):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel = 2,
                 stride = 2,
                 conv_bias = False):
        super(UpsampleConv, self).__init__()
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(input_dim, input_dim, kernel_size=kernel, stride=stride),
            nn.InstanceNorm3d(input_dim),
            nn.PReLU(),
            nn.Conv3d(input_dim, output_dim, 3, 1, 1, bias=conv_bias),
        )
    
    def forward(self, x):
        return self.upsample(x)