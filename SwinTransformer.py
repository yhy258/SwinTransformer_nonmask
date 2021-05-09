import torch.nn as nn
import torch
from models import SwinTransformerBlock, SwinTransformerLayer, PatchMerging, PreProcessing

class SwinTransformer(nn.Module):
    def __init__(self, class_num=100, C=96, num_heads=[3, 6, 12, 24], window_size=7, swin_num_list=[1, 1, 3, 1],
                 norm=True, img_size=224, dropout=0.1, ffn_dim=384):
        super(SwinTransformer, self).__init__()
        self.preprocessing = PreProcessing(hid_dim=C, norm=norm, img_size=img_size)

        features_list = [C, C * 2, C * 4, C * 8]

        stages = nn.ModuleList([])
        stage_layer = SwinTransformerLayer(C=features_list[0], num_heads=num_heads[0], window_size=window_size,
                                           ffn_dim=ffn_dim, act_layer=nn.GELU, dropout=dropout)
        stages.append(SwinTransformerBlock(stage_layer, swin_num_list[0]))
        for i in range(1, 4):
            stages.append(PatchMerging(features_list[i - 1]))
            stage_layer = SwinTransformerLayer(C=features_list[i], num_heads=num_heads[i], window_size=window_size,
                                               ffn_dim=ffn_dim, act_layer=nn.GELU, dropout=dropout)
            stages.append(SwinTransformerBlock(stage_layer, swin_num_list[i]))

        self.stages = stages
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature = features_list[-1]
        self.head = nn.Linear(features_list[-1], class_num)

    def forward(self, x):
        BS, H, W, C = x.size()
        x = self.preprocessing(x)  # BS, L, C
        for stage in self.stages:
            x = stage(x)

        x = x.view(BS, -1, self.feature)

        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x