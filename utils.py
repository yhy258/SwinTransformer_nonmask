import copy
import torch.nn as nn

# shift부분은 attention 넣기 전에 해주자. window 고려도 들어오기 전에..
class SwinAttention(nn.Module):
    def __init__(self, num_heads, C, dropout):
        super().__init__()

        self.scale = C ** -0.5

        self.qkv = nn.Linear(C, C * 3, bias=True)
        self.num_heads = num_heads

        self.softmax = nn.Softmax(dim=-1)

        self.attn_drop = nn.Dropout(0.1)

        self.proj = nn.Linear(C, C)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):  # BS, L, C
        # x = [B, H, W, C]
        B, L, C = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                        4)  # 3, B, Head, L, C_v

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-1, -2))  # dot product

        """
        여기서부터 attention 작업
        """

        attn_score = self.softmax(attn)
        attn_score = self.attn_drop(attn_score)  # L, L
        # B, Head, L, C_v

        out = (attn @ v).transpose(1, 2).flatten(-2)  # B, L, C

        out = self.proj(out)
        out = self.proj_drop(out)

        return out

def window_partition(x, window_size):
    # B, H, W, C : x.size -> B*Window_num, window_size, window_size, C
    B, H, W, C = x.size()
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(x, window_size, H, W):
    # B*Window_num, window_size, window_size, C - > B, H, W, C
    WN = (H//window_size)**2
    B = x.size()[0]//WN
    x = x.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def _clone_layer(layer, num_layers):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])