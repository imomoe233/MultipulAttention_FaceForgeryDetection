import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.inversion_coach import InversionCoach
from components.attention import ChannelAttention, ChannelSpatialAttention, SpatialAttention, DualCrossModalAttention, CoAttention, CrossModalAttention
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from networks.xception import TransferModel

class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""
    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)


class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SRMPixelAttention, self).__init__()
        # self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_srm):
        # x_srm = self.srm(x)
        fea = self.conv(x_srm)        
        att_map = self.pa(fea)
        
        return att_map


class TinySelfAttention(nn.Module):
    '''
    # 使用示例
    input_dim = 128  # 输入特征维度
    hidden_dim = 64  # 注意力中间层维度

    tiny_attention = TinySelfAttention(input_dim, hidden_dim)

    # 生成示例输入
    input_data = torch.randn(16, input_dim)  # 16个样本，每个样本64维特征

    # 使用注意力模块
    output = tiny_attention(input_data)
    print(output.shape)
    '''
    
    def __init__(self, input_dim, hidden_dim):
        super(TinySelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # Calculate query, key, and value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Perform necessary shape transformations
        query = query.view(query.size(0), -1, self.hidden_dim)  # Reshape query
        key = key.view(key.size(0), -1, self.hidden_dim)  # Reshape key
        value = value.view(value.size(0), -1, self.hidden_dim)  # Reshape value
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (self.hidden_dim ** 0.5)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output


class CNNWithAttention(nn.Module):
    def __init__(self, input_channels, input_dim, hidden_dim):
        super(CNNWithAttention, self).__init()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.attention = TinySelfAttention(input_dim, hidden_dim)
        
    def forward(self, x):
        # Apply CNN layers
        cnn_features = self.cnn(x)
        
        # Flatten the feature map
        B, C, H, W = cnn_features.size()
        flattened_features = cnn_features.view(B, C, -1).permute(0, 2, 1)
        
        # Apply attention module
        attention_features = self.attention(flattened_features)
        
        return attention_features


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        #self.csa = ChannelSpatialAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class ConvolutionLayer(nn.Module):
    def __init__(self):
        super(ConvolutionLayer, self).__init__()
        
        # 输入特征的通道数
        in_channels = 14
        
        # 输出特征的通道数
        out_channels = 2048
        
        # 卷积核的大小
        kernel_size = 4
        
        # 步幅
        stride = 8
        
        # 填充
        padding = 0
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        return self.conv(x)


class Two_Stream_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        #self.xception_srm = TransferModel(
        #    'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_lap = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.relu = nn.ReLU(inplace=False)
        '''
        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)

        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)
        '''
        self.coattention0 = CoAttention(in_dim=728, ret_att=False)
        self.coattention1 = CoAttention(in_dim=728, ret_att=False)
        
        self.tiny_attention = TinySelfAttention(input_dim=512, hidden_dim=224)
        #self.tiny_attention = TinySelfAttention(input_dim=512, hidden_dim=512)
        
        self.cross_attention = CrossModalAttention(in_dim=2048)

        self.fusion = FeatureFusionModule()
        self.anglelinear = AngleSimpleLinear(2048, 2)

        self.att_dic = {}
        
        self.coach = InversionCoach()
        print('load encoder.pt successfully pretrained')

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=2048,
            kernel_size=4,  # 调整内核大小以适应分辨率
            stride=8,  # 调整步长以适应分辨率
            padding=0  # 调整填充以适应分辨率
        )
        self.conv_layer = ConvolutionLayer()
        
        # 创建一个全连接层，将输入的通道数从 14 * 224 扩展到 2048 * 15 * 15
        self.fc_layer = nn.Linear(14 * 224, 2048 * 8 * 8)
        #self.fc_layer = nn.Linear(14 * 512, 2048 * 8 * 8)

    def back_features(self, x):
        srm = self.srm_conv0(x)

        x = self.xception_rgb.model.fea_part1_0(x)
        y = self.xception_srm.model.fea_part1_0(srm) \
            + self.srm_conv1(x)
        y = self.relu(y)

        x = self.xception_rgb.model.fea_part1_1(x)
        y = self.xception_srm.model.fea_part1_1(y) \
            + self.srm_conv2(x)
        y = self.relu(y)

        # srm guided spatial attention
        self.att_map = self.srm_sa(srm)
        x = x * self.att_map + x
        x = self.srm_sa_post(x)

        x = self.xception_rgb.model.fea_part2(x)
        y = self.xception_srm.model.fea_part2(y)

        x, y = self.dual_cma0(x, y)


        x = self.xception_rgb.model.fea_part3(x)        
        y = self.xception_srm.model.fea_part3(y)
 

        x, y = self.dual_cma1(x, y)

        x = self.xception_rgb.model.fea_part4(x)
        y = self.xception_srm.model.fea_part4(y)

        x = self.xception_rgb.model.fea_part5(x)
        y = self.xception_srm.model.fea_part5(y)

        fea = self.fusion(x, y)
                
        z = self.tiny_attention(z)
        
        fea = self.cross_attention(z, fea)

        return fea

    def features(self, x, img_lap, encoder_save_path):
        x = x
        #print("load img")
        y = img_lap
        #print("load img_lap")

        z = []
        coach_flag = 0
        
        for i in range(len(encoder_save_path)):
            '''            
            if os.path.exists(encoder_save_path[i]):
                z = torch.stack([z, torch.tensor(torch.load(encoder_save_path[i])[i])], dim=0)
            elif coach_flag == 0:
                z = self.coach.run(x)
                print(z)
                print(type(z))
                print(z[1])
                print(type(z[1]))
                coach_flag = 1
                torch.save(z[i], encoder_save_path[i])
                print("save" + encoder_save_path[i])
            else:
                torch.save(z[i], encoder_save_path[i])
                print("save" + encoder_save_path[i])
             ''' 
             
            if coach_flag == 0:
                z = self.coach.run(x)
                coach_flag = 1
                torch.save(z[i], encoder_save_path[i])
                #print("save" + encoder_save_path[i])
            else:
                torch.save(z[i], encoder_save_path[i])
                #print("save" + encoder_save_path[i])
            
            #z = torch.stack((z), dim=0)


        # x:RGB Stream
        # y:High-frequency Stream
        # z:latent space Stream
        x = self.xception_rgb.model.fea_part1_0(x)
        y = self.xception_lap.model.fea_part1_0(y)
        y = self.relu(y)

        x = self.xception_rgb.model.fea_part1_1(x)
        y = self.xception_lap.model.fea_part1_1(y)
        y = self.relu(y)

        x = self.xception_rgb.model.fea_part2(x)
        y = self.xception_lap.model.fea_part2(y)

        x, y = self.coattention0(x, y)

        x = self.xception_rgb.model.fea_part3(x)        
        y = self.xception_lap.model.fea_part3(y)
 
        x, y = self.coattention1(x, y)

        x = self.xception_rgb.model.fea_part4(x)
        y = self.xception_lap.model.fea_part4(y)

        x = self.xception_rgb.model.fea_part5(x)
        y = self.xception_lap.model.fea_part5(y)

        fea = self.fusion(x, y)
        #print(fea.shape)
                
        z = self.tiny_attention(z)
        #print(z.shape)
        
        # 将输入张量重新形状为适合全连接层的形状
        reshaped_input = z.view(z.size(0), -1)

        # 使用全连接层进行尺寸修改
        z = self.fc_layer(reshaped_input)

        # 将输出张量重新形状为 torch.Size([16, 2048, 15, 15])
        z = z.view(z.size(0), 2048, 8, 8)

        fea = self.cross_attention(fea, z)
        #print(fea.shape)
        
        return fea


    def classifier(self, fea):
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(self, x, img_lap, encoder_save_path):
        '''
        x: original rgb
        
        Return:
        out: (B, 2) the output for loss computing
        fea: (B, 1024) the flattened features before the last FC
        att_map: srm spatial attention map
        '''
        _, fea = self.classifier(self.features(x, img_lap, encoder_save_path))
        out = self.anglelinear(fea)

        return out
    
if __name__ == '__main__':
    # t_list = [transforms.ToTensor()]
    # composed_transform = transforms.Compose(t_list)

    # img = cv2.imread('out.jpg')
    # img = cv2.resize(img, (256, 256))
    # image = composed_transform(img)
    # image = image.unsqueeze(0)

    model = Two_Stream_Net()
    dummy = torch.rand((1,3,256,256))
    out = model(dummy)
    print(out)
    
