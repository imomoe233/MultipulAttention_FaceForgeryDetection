import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Channel Attention and Spaitial Attention from    
Woo, S., Park, J., Lee, J.Y., & Kweon, I. CBAM: Convolutional Block Attention Module. ECCV2018.
"""


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelSpatialAttention, self).__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP_channel = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, 1, 1, bias=False)  # Fix the channel number here
        )

        # Spatial Attention
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)  # Fix the channel number here

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        # Channel Attention
        avgout = self.sharedMLP_channel(self.avg_pool(x))
        maxout = self.sharedMLP_channel(self.max_pool(x))
        channel_attention = self.sigmoid(avgout + maxout)

        # Spatial Attention
        avgout_spatial = torch.mean(x, dim=1, keepdim=True)
        maxout_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avgout_spatial, maxout_spatial], dim=1)
        spatial_attention = self.sigmoid(self.conv1(spatial_features))

        # Combine Channel and Spatial Attention
        attention = channel_attention * spatial_attention

        return x * attention

"""
The following modules are modified based on https://github.com/heykeetae/Self-Attention-GAN
"""


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, out_dim=None, add=False, ratio=8):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.add = add
        if out_dim is None:
            out_dim = in_dim
        self.out_dim = out_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X C X(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, self.out_dim, width, height)

        if self.add:
            out = self.gamma*out + x
        else:
            out = self.gamma*out
        return out  # , attention


class CrossModalAttention(nn.Module):
    """ CMA attention Layer"""

    def __init__(self, in_dim, activation=None, ratio=8, cross_value=True, dropout_prob=0.1, weight_decay=1e-5):
        super(CrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.cross_value = cross_value
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay
        
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H )
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        proj_query = self.query_conv(x).view(
            B, -1, H*W).permute(0, 2, 1)  # B , HW, C
        proj_key = self.key_conv(y).view(
            B, -1, H*W)  # B X C x (*W*H)
        
        # Adding dropout after computing proj_query and proj_key
        proj_query = self.dropout(proj_query)
        proj_key = self.dropout(proj_key)
        
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention = self.softmax(energy)  # BX (N) X (N)
        if self.cross_value:
            proj_value = self.value_conv(y).view(
                B, -1, H*W)  # B , C , HW
        else:
            proj_value = self.value_conv(x).view(
                B, -1, H*W)  # B , C , HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma*out + x

        if self.activation is not None:
            out = self.activation(out)
            
        # L2 regularization
        if self.weight_decay > 0:
            l2_reg = self.weight_decay * sum(p.norm(2) for p in self.parameters())
            out += l2_reg

        return out  # , attention


class DualCrossModalAttention(nn.Module):
    """ Dual CMA attention Layer"""

    def __init__(self, in_dim, activation=None, size=16, ratio=8, ret_att=False):
        super(DualCrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.ret_att = ret_att

        # query conv
        self.key_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv_share = nn.Conv2d(
            in_channels=in_dim//ratio, out_channels=in_dim//ratio, kernel_size=1)

        self.linear1 = nn.Linear(size*size, size*size)
        self.linear2 = nn.Linear(size*size, size*size)

        # separated value conv
        self.value_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.value_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        def _get_att(a, b):
            proj_key1 = self.key_conv_share(self.key_conv1(a)).view(
                B, -1, H*W).permute(0, 2, 1)  # B, HW, C
            proj_key2 = self.key_conv_share(self.key_conv2(b)).view(
                B, -1, H*W)  # B X C x (*W*H)
            energy = torch.bmm(proj_key1, proj_key2)  # B, HW, HW

            attention1 = self.softmax(self.linear1(energy))
            attention2 = self.softmax(self.linear2(
                energy.permute(0, 2, 1)))  # BX (N) X (N)

            return attention1, attention2

        att_y_on_x, att_x_on_y = _get_att(x, y)
        proj_value_y_on_x = self.value_conv2(y).view(
            B, -1, H*W)  # B, C, HW
        out_y_on_x = torch.bmm(proj_value_y_on_x, att_y_on_x.permute(0, 2, 1))
        out_y_on_x = out_y_on_x.view(B, C, H, W)
        out_x = self.gamma1*out_y_on_x + x

        proj_value_x_on_y = self.value_conv1(x).view(
            B, -1, H*W)  # B , C , HW
        out_x_on_y = torch.bmm(proj_value_x_on_y, att_x_on_y.permute(0, 2, 1))
        out_x_on_y = out_x_on_y.view(B, C, H, W)
        out_y = self.gamma2*out_x_on_y + y

        if self.ret_att:
            return out_x, out_y, att_y_on_x, att_x_on_y

        return out_x, out_y  # , attention


class CoAttention(nn.Module):
    def __init__(self, in_dim, activation=None, size=16, ratio=8, ret_att=False):
        super(CoAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.ret_att = ret_att

        # query conv
        self.key_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv_share = nn.Conv2d(
            in_channels=in_dim//ratio, out_channels=in_dim//ratio, kernel_size=1)

        self.linear1 = nn.Linear(size*size, size*size)
        self.linear2 = nn.Linear(size*size, size*size)

        # separated value conv
        self.value_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.value_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
        inputs :
            x : input feature maps( B X C X W X H)
            y : input feature maps( B X C X W X H)
        returns :
            out_x, out_y : co-attention value added feature maps
            [optional attention maps]
        """
        B, C, H, W = x.size()

        # Generate Query and Key for both inputs
        proj_query_x = self.key_conv1(x).view(B, -1, W*H).permute(0, 2, 1) # B x HW x C
        proj_query_y = self.key_conv2(y).view(B, -1, W*H).permute(0, 2, 1) # B x HW x C

        proj_key_x = self.key_conv1(x).view(B, -1, W*H) # B x C x HW
        proj_key_y = self.key_conv2(y).view(B, -1, W*H) # B x C x HW

        # Energy calculation: Energy is calculated differently, taking both inputs into account.
        energy_x = torch.bmm(proj_query_x, proj_key_y) # cross attention for x
        energy_y = torch.bmm(proj_query_y, proj_key_x) # cross attention for y

        # Attention map calculation
        attention_x = self.softmax(energy_x)
        attention_y = self.softmax(energy_y)

        # Apply attention on value
        proj_value_x = self.value_conv1(x).view(B, -1, W*H) # B x C x HW
        proj_value_y = self.value_conv2(y).view(B, -1, W*H) # B x C x HW

        out_x = torch.bmm(proj_value_x, attention_x.permute(0, 2, 1)).view(B, C, H, W)
        out_y = torch.bmm(proj_value_y, attention_y.permute(0, 2, 1)).view(B, C, H, W)

        # Adding original x and y
        out_x = self.gamma1 * out_x + x
        out_y = self.gamma2 * out_y + y

        if self.ret_att:
            return out_x, out_y, attention_x, attention_y

        return out_x, out_y


if __name__ == "__main__":
    x = torch.rand(10, 768, 16, 16)
    y = torch.rand(10, 768, 16, 16)
    dcma = DualCrossModalAttention(768, ret_att=True)
    out_x, out_y, att_y_on_x, att_x_on_y = dcma(x, y)
    print(out_y.size())
    print(att_x_on_y.size())
