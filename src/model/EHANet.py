import torch.nn as nn
import torch
from backbone.mobileVit import mobile_vit_small
from model.blocks import EFAttention, InterlayerEdgeAwareModule, AdaptiveHierarchicalFeatureAggregationModule


class ECNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mobile_vit_small()
        self.iem = InterlayerEdgeAwareModule()

        self.att1 = EFAttention(32)
        self.att2 = EFAttention(64)
        self.att3 = EFAttention(96)
        self.att4 = EFAttention(128)
        self.att5 = EFAttention(160)

        self.d_block_1 = AdaptiveHierarchicalFeatureAggregationModule(160, 128, 128)
        self.d_block_2 = AdaptiveHierarchicalFeatureAggregationModule(128, 96, 96)
        self.d_block_3 = AdaptiveHierarchicalFeatureAggregationModule(96, 64, 64)
        self.d_block_4 = AdaptiveHierarchicalFeatureAggregationModule(64, 32, 32)

        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),  

        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        # encoder
        enc_1, enc_2, enc_3, enc_4, enc_5 = self.encoder(x)

        enc_1 = self.att1(enc_1)
        enc_2 = self.att2(enc_2)
        enc_3 = self.att3(enc_3)
        enc_4 = self.att4(enc_4)
        enc_5 = self.att5(enc_5)

        dec_1, s1 = self.d_block_1(enc_4, enc_5)
        dec_2, s2 = self.d_block_2(enc_3, dec_1)
        dec_3, s3 = self.d_block_3(enc_2, dec_2)
        dec_4, s4 = self.d_block_4(enc_1, dec_3)

        edge_pred, edge_out = self.iem(enc_1, enc_2)
        s_g = self.output_conv(torch.concat([self.up2(dec_4), edge_out], dim=1))

        s1 = self.up16(s1)
        s2 = self.up8(s2)
        s3 = self.up4(s3)
        s4 = self.up2(s4)

        return edge_pred, s1, s2, s3, s4, s_g

    def load_encoder_weight(self):
        self.encoder.load_state_dict(torch.load("backbone/mobilevit_s.pt", map_location=torch.device('cuda')))
