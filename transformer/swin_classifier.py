"""
Swin Transformer Classifier for Hyperspectral Images (84 channels)
------------------------------------------------------------------

This file implements a self-contained Swin-style image classifier with:
- Patch Embedding (Conv2d with stride=patch_size → token sequence)
- Window-based Multi-Head Self-Attention (with Relative Position Bias)
- Shifted Windows (handled by an attention mask built per BasicLayer)
- Hierarchical downsampling via Patch Merging (2x2 tokens → half spatial res, 2x channels)
- Optional spectral dimension reduction (1x1 Conv: 84 -> sr_out_ch)
- Optional "band dropout" (Dropout2d along the spectral/channel axis)

Notes:
- This is a classifier (image-level). For pixel-level tasks use a segmentation head instead.
- Make sure H and W are divisible by patch_size * 2**(num_layers-1) to avoid padding.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# -----------------------------------------------------------------------------
# Window utilities
# -----------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Split a (B, H, W, C) feature map into non-overlapping windows of size (ws x ws).

    Args:
        x: Tensor of shape (B, H, W, C)
        window_size: int, window spatial size (ws)

    Returns:
        windows: Tensor of shape (num_windows * B, ws, ws, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size, window_size,
        W // window_size, window_size,
        C
    )
    # Bring the window dimensions (ws, ws) next to each other and flatten windows into batch
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)     # (B, nH, nW, ws, ws, C)
         .contiguous()
         .view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition: stitch windows back to a (B, H, W, C) feature map.

    Args:
        windows: Tensor of shape (num_windows * B, ws, ws, C)
        window_size: int, ws
        H, W: int, original spatial resolution

    Returns:
        x: Tensor of shape (B, H, W, C)
    """
    # number of windows per image = (H/ws)*(W/ws)
    B = int(windows.shape[0] // (H * W // window_size // window_size))
    x = (
        windows.view(
            B,
            H // window_size, W // window_size,
            window_size, window_size, -1
        )
        .permute(0, 1, 3, 2, 4, 5)      # (B, nH, ws, nW, ws, C)
        .contiguous()
        .view(B, H, W, -1)              # (B, H, W, C)
    )
    return x


# -----------------------------------------------------------------------------
# Feed-Forward Network (MLP)
# -----------------------------------------------------------------------------

class Mlp(nn.Module):
    """
    Simple 2-layer feed-forward network used inside Transformer blocks.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -----------------------------------------------------------------------------
# Window-based Multi-Head Self-Attention with Relative Position Bias
# -----------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """
    Self-Attention computed *locally* inside each window (ws x ws).
    Relative Position Bias encodes pairwise spatial offsets within a window.
    """
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Learnable table of relative position biases for all pairs in a window
        size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, num_heads))

        # Precompute indices that map (i,j) token offsets → row in the bias table
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)                                  # (2, N)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # (N, N, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                          # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        # Linear projection to produce Q, K, V in a single matmul (then split)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B_ , N, C)  where B_ = num_windows * B,  N = ws*ws
            mask: optional attention mask for shifted-windows (shape: [nW, N, N])

        Returns:
            out: (B_, N, C)
        """
        B_, N, C = x.shape

        # Project once → split into Q, K, V for efficiency
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, heads, N, N)

        # Add relative position bias (heads, N, N) → broadcast to batch
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpb = rpb.view(N, N, -1).permute(2, 0, 1).contiguous()  # (heads, N, N)
        attn = attn + rpb.unsqueeze(0)  # (B_, heads, N, N)

        # Mask for shifted windows: prevents tokens from attending across disjoint regions
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # broadcast over batch and heads
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Aggregate values and project back
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------------------------------------------------------
# Patch Embedding
# -----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Convert an image (B, C, H, W) into a token sequence (B, L, embed_dim):
    - A Conv2d with kernel=stride=patch_size extracts non-overlapping patches.
    - Each patch is linearly projected to 'embed_dim' channels.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0],
                                   img_size[1] // patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        # (B, C, H, W) → (B, embed_dim, H', W')
        x = self.proj(x)
        # Flatten spatial dims and bring tokens to sequence dimension
        x = x.flatten(2).transpose(1, 2)  # (B, L, embed_dim)
        if self.norm is not None:
            x = self.norm(x)
        return x


# -----------------------------------------------------------------------------
# Patch Merging (Downsample by 2 in H and W, double channels)
# -----------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """
    Merge 2x2 neighboring tokens:
    - Input:  (B, L, C) with spatial resolution (H, W) where L=H*W
    - Output: (B, L/4, 2C)
    Steps:
      1) Reshape to (B, H, W, C)
      2) Pick pixels at (even,even), (odd,even), (even,odd), (odd,odd)
      3) Concatenate along channel → (B, H/2, W/2, 4C)
      4) Linear reduction to 2C
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "PatchMerging: input L must match H*W"
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # even rows, even cols
        x1 = x[:, 1::2, 0::2, :]  # odd rows,  even cols
        x2 = x[:, 0::2, 1::2, :]  # even rows, odd cols
        x3 = x[:, 1::2, 1::2, :]  # odd rows,  odd cols

        x = torch.cat([x0, x1, x2, x3], dim=-1)       # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)                      # (B, L/4, 4C)
        x = self.norm(x)
        x = self.reduction(x)                         # (B, L/4, 2C)
        return x


# -----------------------------------------------------------------------------
# Swin Transformer Block (W-MSA / SW-MSA + MLP)
# -----------------------------------------------------------------------------

class SwinTransformerBlock(nn.Module):
    """
    A single Swin block:
      - LayerNorm
      - (Shifted) Window Attention
      - Residual connection
      - LayerNorm + MLP
      - Residual connection
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        # Cap window to the current spatial size if needed
        H, W = input_resolution
        self.window_size = min(window_size, H, W)
        # Only shift if the window actually fits inside the feature map
        self.shift_size = 0 if min(H, W) <= self.window_size else shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        self.H, self.W = input_resolution

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "SwinBlock: input L must match H*W"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Optionally roll the feature map to implement "shifted windows"
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x

        # Partition into windows, run attention inside each window, then reverse
        x_windows = window_partition(shifted, self.window_size)                 # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, N, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)                     # (nW*B, N, C)

        # Stitch windows back and undo the roll
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted = window_reverse(attn_windows, self.window_size, H, W)          # (B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted

        x = x.view(B, H * W, C)

        # Residual + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -----------------------------------------------------------------------------
# BasicLayer: stack of Swin blocks (+ optional PatchMerging)
# -----------------------------------------------------------------------------

class BasicLayer(nn.Module):
    """
    A hierarchical stage:
      - Repeats SwinTransformerBlock 'depth' times
      - Uses an attention mask shared across shifted blocks within the stage
      - Optionally downsamples at the end via PatchMerging
    """
    def __init__(self, dim, input_resolution, depth,
                 num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Allow a per-block drop_path schedule (list) or a single scalar
        if isinstance(drop_path, (list, tuple)):
            dpr = list(drop_path)
        else:
            dpr = [drop_path] * depth

        # Build the block stack; odd blocks use shift_size = window_size//2
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=dpr[i],
                    norm_layer=norm_layer
                )
            )

        # Optional downsampling at the end of the stage
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) \
            if downsample is not None else None

        # Build the attention mask used by shifted-window blocks in this stage
        self.attn_mask = self._create_attn_mask(window_size)

    def _create_attn_mask(self, window_size: int):
        """
        Create an attention mask for shifted window attention:
        tokens rolled from different regions shouldn't attend to each other.
        """
        H, W = self.input_resolution
        if min(H, W) <= window_size:
            # No meaningful shift possible → no mask needed
            return None

        img_mask = torch.zeros((1, H, W, 1))  # a grid of region IDs
        cnt = 0
        ws = window_size
        ss = window_size // 2

        # Three vertical and three horizontal slices produce 9 regions
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Partition the labeled image into windows and compute pairwise differences
        mask_windows = window_partition(img_mask, ws)  # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, ws * ws)  # (nW, N)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        attn_mask = self.attn_mask.to(x.device) if self.attn_mask is not None else None

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


# -----------------------------------------------------------------------------
# Swin Transformer Classifier (Hyperspectral)
# -----------------------------------------------------------------------------

class SwinTransformerClassifierHSI(nn.Module):
    """
    Image-level classifier for hyperspectral images.

    Key options:
      - spectral_reduction: optional 1x1 Conv to reduce spectral channels, e.g., 84 → 32
      - band_dropout_p: Dropout2d on the channel/spectral axis (acts like randomly dropping bands)
      - ape: absolute positional embedding (not required because Swin uses relative bias)

    Forward flow:
      HSI (B, C, H, W)
        → optional spectral reduction (1x1)
        → optional band dropout
        → PatchEmbed → sequence tokens (B, L, C)
        → [BasicLayer x num_layers] with shifted windows + patch merging
        → LayerNorm
        → AdaptiveAvgPool1d over tokens (global context)
        → Linear head → logits (B, num_classes)
    """
    def __init__(self, img_size=128, patch_size=4, in_chans=84, num_classes=10,
                 embed_dim=128, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,
                 spectral_reduction=False, sr_out_ch=32, band_dropout_p=0.0):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # channel dim before the head
        self.mlp_ratio = mlp_ratio
        self.spectral_reduction = spectral_reduction

        # Optional spectral reduction: 1x1 conv to reduce channel/band count
        if spectral_reduction:
            self.sr = nn.Conv2d(in_chans, sr_out_ch, kernel_size=1, bias=False)
            in_chans = sr_out_ch
        else:
            self.sr = None

        # Optional "band dropout": Dropout2d randomly zeroes entire channels per-sample
        self.band_dropout = nn.Dropout2d(p=band_dropout_p) if band_dropout_p > 0 else nn.Identity()

        # Patch embedding: Conv2d stride=patch_size → (B, L, embed_dim)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Optional absolute positional embedding (Swin mainly uses relative bias)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth schedule across all blocks in all layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build hierarchical stages (BasicLayer)
        self.layers = nn.ModuleList()
        dp_idx = 0
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[dp_idx: dp_idx + depths[i_layer]],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            dp_idx += depths[i_layer]
            self.layers.append(layer)

        # Normalization before pooling & classification head
        self.norm = norm_layer(self.num_features)

        # Global pooling over the token dimension → a single vector per image
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Final classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=84, H, W)
        if self.sr is not None:
            x = self.sr(x)            # optional spectral reduction: (B, sr_out_ch, H, W)
        x = self.band_dropout(x)      # optional band-wise dropout regularization

        x = self.patch_embed(x)       # (B, L, embed_dim)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Hierarchical stages
        for layer in self.layers:
            x = layer(x)              # spatial downsampling happens inside the layer (except last)

        # Normalize, pool over tokens, and flatten
        x = self.norm(x)              # (B, L, C_last)
        x = self.avgpool(x.transpose(1, 2))  # (B, C_last, 1)
        x = torch.flatten(x, 1)       # (B, C_last)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)              # (B, num_classes)
        return x


# -----------------------------------------------------------------------------
# Quick sanity test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: hyperspectral image 128x128 with 84 bands
    B, C, H, W = 2, 84, 128, 128
    x = torch.randn(B, C, H, W)

    # Baseline: just set in_chans=84
    model_base = SwinTransformerClassifierHSI(
        img_size=128, patch_size=4, in_chans=84, num_classes=5,
        embed_dim=128, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
        window_size=8, spectral_reduction=False
    )
    y_base = model_base(x)
    print("baseline logits:", y_base.shape)  # -> (B, 5)

    # With spectral reduction 84 -> 32 and small band dropout
    model_sr = SwinTransformerClassifierHSI(
        img_size=128, patch_size=4, in_chans=84, num_classes=5,
        embed_dim=128, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
        window_size=8, spectral_reduction=True, sr_out_ch=32, band_dropout_p=0.05
    )
    y_sr = model_sr(x)
    print("spectral-reduced logits:", y_sr.shape)  # -> (B, 5)
