import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
""" Squeeze and Excitation block """
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

""" Adaptive Hierarchical Feature Aggregation Module """
class AdaptiveHierarchicalFeatureAggregationModule(nn.Module):
    def __init__(self, in_channels, low_channels, high_channels):
        super(AdaptiveHierarchicalFeatureAggregationModule, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, low_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.conv_b1 = nn.Sequential(
            nn.Conv2d(2*low_channels, high_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(high_channels),
            SELayer(high_channels, high_channels)
        )

        self.conv_b2 = nn.Sequential(
            nn.Conv2d(2*low_channels, high_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_channels, high_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(high_channels),
        )

        self.conv_rp = nn.Sequential(
            nn.Conv2d(high_channels, high_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True),
        )

        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(high_channels, high_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_channels, high_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(high_channels),
            nn.Sigmoid()
        )

        self.out = nn.Sequential(
            nn.Conv2d(high_channels, 1, kernel_size=1, bias=False),
        )


    def forward(self, x_p, x_d):
        x_concat = torch.cat((x_p, self.input_conv(x_d)), 1)
        d = self.conv_b1(x_concat)+self.conv_b2(x_concat)
        rp = self.out(self.conv_rp(d)*self.weight(d))

        return d, rp

''' Efficient Fusion Attention Module '''
class EFAttention(nn.Module):
    def __init__(self, in_channels, kernel_size = 3):
        super().__init__()

        # x_c
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

        # x_s
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,x):
        y1 = self.gap(x) 
        y1 = y1.squeeze(-1).permute(0, 2, 1)  
        y1 = self.conv(y1)  
        y1 = self.sigmoid(y1)  
        y1 = y1.permute(0, 2, 1).unsqueeze(-1)  
        x_c =  x * y1.expand_as(x)

        q = self.Conv1x1(x)  
        q = self.norm(q)
        x_s = x * q  

        return x_c + x_s


class InterlayerEdgeAwareModule(nn.Module):
    """ Inter-layer Edge-aware Module  """

    def __init__(self):
        super().__init__()
        layer1_channels = 32
        layer2_channels = 64
        out_edge_channels = 2
        out_channels = 32

        self.input_conv_1 = nn.Sequential(
            nn.Conv2d(layer1_channels, out_edge_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_edge_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_edge_channels, out_edge_channels, kernel_size=3, padding=1, bias=False),  # 3x3
            nn.BatchNorm2d(out_edge_channels),
            nn.ReLU(inplace=True),
        )

        self.input_conv_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(layer2_channels, out_edge_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_edge_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_edge_channels, out_edge_channels, kernel_size=3, padding=1, bias=False),  # 3x3
            nn.BatchNorm2d(out_edge_channels),
            nn.ReLU(inplace=True),
        )

        self.output_edge_conv = nn.Sequential(
            nn.Conv2d(2 * out_edge_channels, out_edge_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_edge_channels),
            nn.ReLU(inplace=True),  # No ReLU because of incoming softmax

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(out_edge_channels, out_edge_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_edge_channels),
            nn.Softmax(dim=1)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(2 * out_edge_channels, out_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_1, x_2):
        input_conv_1 = self.input_conv_1(x_1)
        input_conv_2 = self.input_conv_2(x_2)
        input_conv = torch.cat((input_conv_1, input_conv_2), dim=1)
        return self.output_edge_conv(input_conv), self.output_conv(input_conv)
