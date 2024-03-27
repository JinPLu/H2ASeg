import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

class Adaptive_Weighting(nn.Module):
    """
    MutiModle-Channel Emphasize Attention 
    return: Enphasize(concat(ct, pet, f_hi))
    """
    def __init__(self, 
                 input_channels1,
                 input_channels2,
                 input_size,
                 scale = 4):
        super(Adaptive_Weighting, self).__init__()
        
        self.input_channels1 = input_channels1
        self.input_channels2 = input_channels2
        self.input_channels = input_channels1 * 2 + input_channels2 * 2
        self.globalAvgPool = nn.AvgPool3d(kernel_size=tuple(input_size))
        self.fc1 = nn.Linear(in_features=self.input_channels         , out_features=self.input_channels // scale)
        self.fc2 = nn.Linear(in_features=self.input_channels // scale, out_features=self.input_channels         )
        self.tanh = nn.Tanh()

    def forward(self, ct, pet, f_hi):
        b, c = ct.size()[:2]
        original_ct = ct
        original_pet = pet
        original_f_hi = f_hi

        # out_ct/out_pet:(b, c1, 1, 1, 1)
        # out_f_hi: (b, c2, 1, 1, 1)
        ct = self.globalAvgPool(ct)
        pet = self.globalAvgPool(pet)
        f_hi = self.globalAvgPool(f_hi)
        
        ct = ct.view(b, -1)
        pet = pet.view(b, -1)
        f_hi = f_hi.view(b, -1)
        # out:(b, 2*c1 + c2)
        f = torch.concat([ct, pet, f_hi], dim=1)

        # channel attention
        weight = self.fc1(f)
        weight = self.tanh(weight)
        weight = self.fc2(weight)
        weight = self.tanh(weight)
        
        weight = weight.view(b, self.input_channels, 1, 1, 1)
        ct   = weight[:, 0  :c  ] * original_ct
        pet  = weight[:, c  :2*c] * original_pet
        f_hi = weight[:, 2*c:   ] * original_f_hi
        return ct, pet, f_hi
    
class UniBidirectional_Spatial_Attention(nn.Module):
    '''
    UniBidirectional_Spatial_Attention
    '''

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout = 0.25,
                 kernel_size = 1,
                 padding = 0,
                 channel_scale = 1,
                 conv_bias = False,
                 skip_connection=False):
        super(UniBidirectional_Spatial_Attention, self).__init__()

        self.channel_in = in_dim
        self.channel_out = out_dim
        self.channel_scale = channel_scale
        self.conv_bias = conv_bias
        self.skip = skip_connection
        
        self.Q_conv_target_target = nn.Conv3d(self.channel_in, self.channel_in // self.channel_scale, kernel_size=kernel_size,
                                    bias=self.conv_bias, padding=padding)
        self.Q_conv_source_target = nn.Conv3d(self.channel_in, self.channel_in // self.channel_scale, kernel_size=kernel_size,
                                    bias=self.conv_bias, padding=padding)
        self.K_conv_target_target = nn.Conv3d(self.channel_in, self.channel_in // self.channel_scale, kernel_size=kernel_size,
                                    bias=self.conv_bias, padding=padding)
        self.K_conv_source_target = nn.Conv3d(self.channel_in, self.channel_in // self.channel_scale, kernel_size=kernel_size,
                                    bias=self.conv_bias, padding=padding)
        self.V_conv_target_target = nn.Conv3d(self.channel_in, self.channel_in, kernel_size=kernel_size, 
                                    bias=self.conv_bias, padding=padding)
        self.V_conv_source_target = nn.Conv3d(self.channel_in, self.channel_in, kernel_size=kernel_size, 
                                    bias=self.conv_bias, padding=padding)
        self.out = nn.Conv3d(self.channel_in, self.channel_out, kernel_size=kernel_size, bias=self.conv_bias,
                                padding=padding)

        self.attn_dropout_target_target = nn.Dropout(dropout)
        self.attn_dropout_source_target = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, source, target):
        b, c, h, w, d = source.shape
        # (Nh Nw Nd)
        q_target_target = self.Q_conv_target_target(target)
        q_source_target = self.Q_conv_source_target(target)
        
        k_target_target = self.K_conv_target_target(target)
        k_source_target = self.K_conv_source_target(source)
        
        v_target_target = self.V_conv_target_target(target)
        v_source_target = self.V_conv_source_target(source)

        q_target_target = rearrange(q_target_target, 'b c h w d -> b (h w d) c')
        q_source_target = rearrange(q_source_target, 'b c h w d -> b (h w d) c')
        k_target_target = rearrange(k_target_target, 'b c h w d -> b c (h w d)')
        k_source_target = rearrange(k_source_target, 'b c h w d -> b c (h w d)')
        v_target_target = rearrange(v_target_target, 'b c h w d -> b c (h w d)')
        v_source_target = rearrange(v_source_target, 'b c h w d -> b c (h w d)')

        scores_target_target = torch.einsum('bmx, bxn -> bmn', [q_target_target, k_target_target])
        scores_source_target = torch.einsum('bmx, bxn -> bmn', [q_source_target, k_source_target])

        sqrt_dim = (c // self.channel_scale) ** 0.5
        target_target_weights = scores_target_target / sqrt_dim
        target_target_weights = self.softmax(target_target_weights)
        target_target_weights = self.attn_dropout_target_target(target_target_weights)

        weights_source_target = scores_source_target / sqrt_dim
        weights_source_target = self.softmax(weights_source_target)
        weights_source_target = self.attn_dropout_source_target(weights_source_target)

        target_target_attention = torch.einsum('bmx, bnx -> bmn', [target_target_weights, v_target_target])
        target_target_attention = rearrange(target_target_attention, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)

        source_target_attention = torch.einsum('bmx, bnx -> bmn', [weights_source_target, v_source_target])
        source_target_attention = rearrange(source_target_attention, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)

        attention = self.out((target_target_attention + source_target_attention) / 2)
        attention = self.out_dropout(attention)
        if self.skip:
            attention = attention + target
        return attention

class Bidirectional_Spatial_Attention(nn.Module):
    '''
    Bidirectional_Spatial_Attention
    '''

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout = 0.25,
                 kernel_size = 1,
                 padding = 0,
                 channel_scale = 1,
                 conv_bias = False,
                 skip_connection=False):
        super(Bidirectional_Spatial_Attention, self).__init__()
        self.sa_ct = UniBidirectional_Spatial_Attention(in_dim, out_dim, dropout,
                                                        kernel_size, padding, channel_scale,
                                                        conv_bias, skip_connection)
        self.sa_pet = UniBidirectional_Spatial_Attention(in_dim, out_dim, dropout,
                                                        kernel_size, padding, channel_scale,
                                                        conv_bias, skip_connection)
    def forward(self, ct, pet):
        att_ct = self.sa_ct(pet, ct)
        att_pet = self.sa_pet(ct, pet)
        return att_ct, att_pet    

class Intra_Windows_Spatial_Attention(nn.Module):
    '''
    MutiModal Intra-Windows Spatial Attention

    '''
    def __init__(self,
                 in_dim,
                 out_dim,
                 window_size,
                 dropout = 0.25,
                 kernel_size = 1,
                 padding = 0,
                 channel_scale = 1,
                 conv_bias = False):
        super(Intra_Windows_Spatial_Attention, self).__init__()

        self.window_size = window_size
        self.sa = Bidirectional_Spatial_Attention(in_dim, out_dim, dropout,
                                               kernel_size, padding, channel_scale,
                                               conv_bias, skip_connection=False)

    def forward(self, ct, pet):
        b, c, h, w, d = ct.shape
        Wh, Ww, Wd = self.window_size
        Nh, Nw, Nd = h // Wh, w // Ww, d // Wd

        # (Nh Nw Nd)
        ct_wins = rearrange(ct, 'b c (Nh Wh) (Nw Ww) (Nd Wd) -> (b Nh Nw Nd) c Wh Ww Wd', Wh=Wh, Ww=Ww, Wd=Wd)
        pet_wins = rearrange(pet, 'b c (Nh Wh) (Nw Ww) (Nd Wd) -> (b Nh Nw Nd) c Wh Ww Wd', Wh=Wh, Ww=Ww, Wd=Wd)
        
        ct_attention, pet_attention = self.sa(ct_wins, pet_wins)
        ct_attention = rearrange(ct_attention, '(b Nh Nw Nd) c Wh Ww Wd  -> b c (Nh Wh) (Nw Ww) (Nd Wd)',
                                 b=b, Wh=Wh, Ww=Ww, Wd=Wd, Nh=Nh, Nw=Nw, Nd=Nd)
        pet_attention = rearrange(pet_attention, '(b Nh Nw Nd) c Wh Ww Wd  -> b c (Nh Wh) (Nw Ww) (Nd Wd)',
                                 b=b, Wh=Wh, Ww=Ww, Wd=Wd, Nh=Nh, Nw=Nw, Nd=Nd)
        return ct_attention, pet_attention
    

class MCSA(nn.Module):
    '''
    Cross-Modal Cross-Window Spatial Attention
    input: ct_feature, pet_feature
    return: ct_attention, pet_attention
    '''

    def __init__(self,
                 in_dim,
                 # input_size,
                 window_size = [8,8,4],
                 dropout = 0.25,
                 channel_scale = 4,
                 activation = nn.PReLU,
                 conv_bias = False):
        super(MCSA, self).__init__()
        self.channel_in = in_dim
        # self.input_size = input_size
        self.window_size = window_size
        self.channel_scale = channel_scale
        self.dropout = dropout
        self.activation = activation
        self.conv_bias = conv_bias
        
        self.pool_size = int(window_size.min())
        self.win_merge_ct = nn.Conv3d(self.channel_in, self.channel_in, kernel_size=self.pool_size, stride=self.pool_size)
        self.win_merge_pet = nn.Conv3d(self.channel_in, self.channel_in, kernel_size=self.pool_size, stride=self.pool_size)
        # self.avg = nn.AvgPool3d(kernel_size=tuple(window_size))

        self.inter_window_attention = Bidirectional_Spatial_Attention(in_dim=self.channel_in,
                                                                      out_dim=self.channel_in,
                                                                      dropout=self.dropout,
                                                                      channel_scale=self.channel_scale,
                                                                      conv_bias=self.conv_bias)
        self.upscale = nn.Upsample(scale_factor=self.pool_size, mode="trilinear", align_corners=True)
        
        self.intra_window_attention = Intra_Windows_Spatial_Attention(in_dim=self.channel_in,
                                                                      out_dim=self.channel_in,
                                                                      window_size=self.window_size,
                                                                      dropout=self.dropout,
                                                                      channel_scale=self.channel_scale,
                                                                      conv_bias=self.conv_bias)  
        self.IR_ct1 = self.norm_lrelu(self.channel_in)
        self.IR_ct2 = self.norm_lrelu(self.channel_in)
        self.IR_pet1 = self.norm_lrelu(self.channel_in)
        self.IR_pet2 = self.norm_lrelu(self.channel_in)
        
    def norm_lrelu(self, feat):
        return nn.Sequential(
            nn.InstanceNorm3d(feat),
            self.activation(),
        )

    def forward(self, ct, pet):
        '''
        skip-connection:
        original feature -> intra-window attention
        intra-window attention -> inter-window attention(expand)
        '''
        ct_attention1 = self.win_merge_ct(ct)
        pet_attention1 = self.win_merge_pet(pet)
        ct_attention1, pet_attention1 = self.inter_window_attention(ct_attention1, pet_attention1)
        ct_attention1 = self.IR_ct1(self.upscale(ct_attention1) + ct)
        pet_attention1 = self.IR_pet1(self.upscale(pet_attention1) + pet)
        
        ct_attention2, pet_attention2 = self.intra_window_attention(ct_attention1, pet_attention1)
        ct_attention = self.IR_ct2(ct_attention2 + ct_attention1)
        pet_attention = self.IR_pet2(pet_attention2 + pet_attention1)

        return ct_attention, pet_attention

