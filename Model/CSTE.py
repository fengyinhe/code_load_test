import torch
from torch import nn

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import typing as t
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models.resnet import ResNet50_Weights
from math import sqrt
import gc
# helpers
feat_dim = 384
out_channels = 1

def exists(val):
    return val is not None


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class DINO_EXP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class, cnn_kwargs=None):
        super(DINO_EXP, self).__init__()
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        self.cnn = ResNet50(**cnn_kwargs)
        self.conv_fuse = nn.Conv2d(704, 512, kernel_size=1)
        self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dinov2_vits14.eval()
        self.SET = SET(512)
        self.ffa = FFA(dim=512, head_num=8)
        self.setvit = STEViT(
            image_size=224,  # image size
            frames=64,  # number of frames
            image_patch_size=14,  # image patch size
            frame_patch_size=2,  # frame patch size
            num_classes=1,
            dim=1024,
            spatial_depth=6,  # depth of the spatial transformer
            temporal_depth=6,  # depth of the temporal transformer
            heads=8,
            mlp_dim=2048,
            variant='Spatiotemporal_attention',  # or 'Global Attention'
        )
        self.fc = nn.Linear(384*32, 32)

        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)

    def fuse_features(self, features):

        target_size = features[16].size()[2:]


        fused_features = []
        for k in [2, 4, 16]:
            feat = features[k]
            feat_up = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            fused_features.append(feat_up)

        # concat
        fused = torch.cat(fused_features, dim=1)
        del features
        del fused_features

        fused = self.conv_fuse(fused)
        return fused

    def forward(self, x):
        B, C, T, H, W = x.shape    # (2,3,64,224,224)

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # shape (B*T, C, H, W)(2*64,3,224,224)
        feature_pyramid = self.cnn(x)
        features_dict = self.dinov2_vits14.forward_features(x)  # DINOv2
        x = features_dict['x_norm_patchtokens']  # shape(128,256,384)
        x = rearrange(x, 'B (H W) D -> B D H W', H=9)
        del features_dict

        # ff
        feature_pyramid[16] = x
        x = self.fuse_features(feature_pyramid)
        del feature_pyramid
        x = rearrange(x, 'B D H W -> B (H W) D')
        x = rearrange(x, '(B T) HW C -> B C T HW', B=B)
        x = self.ffa(x)

        # STETansformer
        x = rearrange(x, 'B D (T PT) N ->B T N (D PT)', PT=2)
        x = self.setvit(x)

        x = rearrange(x, 'B T D -> B D T', D=1024)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)  # shape(batch_size, n_class)

        return out


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res=False, weights=None,
                 dilation=None, freeze_bn=True, anti_aliased=False, early_exit=False, amp=False,
                 amp_dtype=torch.float16) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False, False, False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights=weights, replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(weights=ResNet50_Weights.DEFAULT, replace_stride_with_dilation=dilation)

        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        net = self.net
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        Thefeats = {2: x}
        x = net.maxpool(x)
        x = net.layer1(x)
        Thefeats[4] = x
        del x
        return Thefeats



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn = TET()
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    @staticmethod
    def generate_square_subsequent_mask(size):

        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        B, N, D = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads), qkv)
        attn_mask = None
        output, _ = self.attn(q, k, v, attn_mask)

        out = rearrange(output, 'b n h d -> b n (h d)')

        return self.to_out(out)


class TET(nn.Module):
    '''De-stationary Attention'''
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0., output_attention=False):
        super(TET, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S

        # TET Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag and attn_mask is not None:
            # 直接使用attn_mask进行掩码操作
            scores.masked_fill_(attn_mask == float('-inf'), float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class Spatial_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SET(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            att = rearrange(x, '(b f) n d -> b d f n', f=32)
            att = attn(att)
            att = rearrange(att, 'b d f n -> (b f) n d')
            x = att + x
            del att
            x = ff(x) + x
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class FactorizedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        b, f, n, _ = x.shape
        for spatial_attn, temporal_attn, ff in self.layers:
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = spatial_attn(x) + x
            x = rearrange(x, '(b f) n d -> (b n) f d', b=b, f=f)
            x = temporal_attn(x) + x
            x = ff(x) + x
            x = rearrange(x, '(b n) f d -> b f n d', b=b, n=n)

        return self.norm(x)


class STEViT(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            image_patch_size,
            frames,
            frame_patch_size,
            num_classes,
            dim,
            spatial_depth,
            temporal_depth,
            heads,
            mlp_dim,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.,
            variant='factorized_encoder',
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        assert variant in ('factorized_encoder', 'factorized_self_attention'), f'variant = {variant} is not implemented'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        if variant == 'Spatiotemporal_attention':
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
            self.spatial_transformer = Spatial_Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
            self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)
        elif variant == 'Global_Attention':
            assert spatial_depth == temporal_depth, 'Spatial and temporal depth must be the same for Global_Attention'
            self.factorized_transformer = FactorizedTransformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        self.variant = variant

    def forward(self, video):

        x = video
        b, f, n, _ = x.shape

        x = x + self.pos_embedding[:, :f, :n]
        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b=b, f=f)
            x = torch.cat((spatial_cls_tokens, x), dim=2)

        x = self.dropout(x)

        if self.variant == 'Spatiotemporal_attention':
            x = rearrange(x, 'b f n d -> (b f) n d')

            x = self.spatial_transformer(x)

            x = rearrange(x, '(b f) n d -> b f n d', b=b)
            # excise out the spatial cls tokens or average pool for temporal attention

            x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')
            # append temporal CLS tokens

            if exists(self.temporal_cls_token):
                temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b=b)

                x = torch.cat((temporal_cls_tokens, x), dim=1)

            # attend across time

            x = self.temporal_transformer(x)

            # # excise out temporal cls token or average pool
            #
            # x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        elif self.variant == 'Global_Attention':
            x = self.factorized_transformer(x)
            x = x[:, 0, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b d', 'mean')

        x = self.to_latent(x)
        return x

class FFA(nn.Module):

    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(FFA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode


        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The dim of x is (B, C, H, W)
        """
        # Channel attention based on self attention
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        fft_x = torch.fft.fft2(x)
        x = torch.fft.ifft2(fft_x).real
        x = self.conv1(x)
        return self.sigmoid(x)


class SET(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(SET, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result
