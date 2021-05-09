import math
import torch
import torch.nn as nn
from utils import _clone_layer, window_partition, window_reverse, SwinAttention

class PreProcessing(nn.Module):  # patch partition, embedding,
    def __init__(self, hid_dim=96, norm=True, img_size=224):
        super().__init__()
        self.embed = nn.Conv2d(3, hid_dim, kernel_size=4, stride=4)
        self.norm_layer = None
        self.norm = norm
        if self.norm:
            self.norm_layer = nn.LayerNorm(hid_dim)

        self.num_patches = img_size // 4

        self.hid_dim = hid_dim

    def forward(self, x):
        BS, H, W, C = x.size()

        x = self.embed(x).flatten(2).transpose(1, 2)  # BS, C, L -> BS, L, C

        if self.norm:
            self.norm_layer(x)

        return x  # [Bs, L, C]

class SwinTransformerBlock(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = _clone_layer(layer, num_layers)

    def forward(self ,x):
        for layer in self.layers:
            x = layer(x)

        return x


class SwinTransformerLayer(nn.Module):
    def __init__(self, C, num_heads, window_size, ffn_dim, act_layer=nn.GELU, dropout=0.1):
        super().__init__()
        self.mlp1 = Mlp(C, ffn_dim, act_layer=nn.GELU, drop=dropout)
        self.mlp2 = Mlp(C, ffn_dim, act_layer=nn.GELU, drop=dropout)

        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.norm3 = nn.LayerNorm(C)
        self.norm4 = nn.LayerNorm(C)

        self.shift_size = window_size // 2
        self.window_size = window_size
        self.W_MSA = SwinAttention(num_heads=num_heads, C=C, dropout=dropout)
        self.SW_MSA = SwinAttention(num_heads=num_heads, C=C, dropout=dropout)

    def forward(self, x):  # BS, L, C
        BS, L, C = x.shape
        S = int(math.sqrt(L))

        shortcut = x

        x = self.norm1(x)  # BS, L, C

        x_windows = self.window_to_attention(x, S, C)

        attn_x = self.W_MSA(x_windows)

        x = self.attention_to_og(attn_x, S, C)

        x = x + shortcut

        shorcut = x

        x = self.norm2(x)
        x = self.mlp1(x)

        x = x + shortcut

        shortcut = x

        x = self.norm3(x)

        x_windows = self.window_to_attention(x, S, C, shift=True)

        x_attn = self.SW_MSA(x_windows)

        x = self.attention_to_og(x, S, C, shift=True)

        x = x + shortcut

        shortcut = x

        x = self.norm4(x)
        x = self.mlp2(x)

        return x + shortcut

    def window_to_attention(self, x, S, C, shift=False):
        x = x.view(-1, S, S, C)
        if shift:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        return x_windows

    def attention_to_og(self, attn_x, S, C, shift=False):
        attn_x = attn_x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_x, self.window_size, S, S)
        if shift:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(-1, S * S, C)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self,C):
        super().__init__()
        self.proj = nn.Linear(C*4, C*2)

    def forward(self ,x): # BS,H,W,C
        BS, L, C = x.size()
        H = int(math.sqrt(L))
        x = x.view(BS, H//2, H//2, C*4)
        x = self.proj(x)
        x = x.view(BS, -1, C*2)
        return x