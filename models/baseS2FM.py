

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from VMamba.classification.models.vmamba import VSSBlock, LayerNorm2d
    from mamba_ssm import Mamba
    from models.vmamba import CVSSDecoderBlock
    from models.ghostnet import GhostModule
    from models.se_module import SELayer
except ImportError:
    print("警告: 无法从项目中导入部分模块，将使用 nn.Module 作为占位符。")
    VSSBlock, LayerNorm2d, Mamba, CVSSDecoderBlock, GhostModule, SELayer = [nn.Module] * 6


# ===================================================================
# SECTION 1: S2FM 模块及其辅助函数 (来自您最初的 baseS2FM.py)
# ===================================================================

def concat_features(FT1: torch.Tensor, FT2: torch.Tensor) -> torch.Tensor:
    return torch.cat([FT1, FT2], dim=1)


def four_way_scan(X: torch.Tensor) -> List[torch.Tensor]:
    B, C, H, W = X.shape;
    L = H * W
    X_lr = X.permute(0, 2, 3, 1).reshape(B, L, C)
    X_rl = torch.flip(X, dims=[3]).permute(0, 2, 3, 1).reshape(B, L, C)
    X_ud = X.permute(0, 3, 2, 1).reshape(B, L, C)
    X_du = torch.flip(X, dims=[2]).permute(0, 3, 2, 1).reshape(B, L, C)
    return [X_lr, X_rl, X_ud, X_du]


def mamba_ssm_scan(X_lr, X_rl, X_ud, X_du, mamba_block):
    device = mamba_block.out_proj.weight.device
    X_lr, X_rl, X_ud, X_du = X_lr.to(device), X_rl.to(device), X_ud.to(device), X_du.to(device)
    Y_lr = mamba_block(X_lr)
    Y_rl = mamba_block(X_rl)
    Y_ud = mamba_block(X_ud)
    Y_du = mamba_block(X_du)
    return Y_lr, Y_rl, Y_ud, Y_du


class GatedResidual(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.linear_pi = nn.Linear(2 * num_channels, num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_list, y_list):
        y_gated_list = []
        for x, y in zip(x_list, y_list):
            pi = self.sigmoid(self.linear_pi(torch.cat([x, y], dim=2)))
            y_gated = y * pi + x * (1 - pi)
            y_gated_list.append(y_gated)
        return y_gated_list


def four_way_inverse_scan(four_seq: List[torch.Tensor], H: int, W: int) -> torch.Tensor:
    shape = four_seq[0].shape;
    B, L, C = shape
    y_sum = torch.sum(torch.stack(four_seq), dim=0)
    return y_sum.reshape(B, H, W, C).permute(0, 3, 1, 2)


class S2FM(nn.Module):
    def __init__(self, num_channels: int, r1: int = 4, r2: int = 8, **kwargs):
        super().__init__()
        input_ch = 2 * num_channels;
        upscaled_ch = input_ch * r1;
        mamba_dim = upscaled_ch // r2
        self.pre_mamba_proj = nn.Sequential(
            GhostModule(inp=input_ch, oup=upscaled_ch, relu=True),
            SELayer(channel=upscaled_ch),
            nn.Conv2d(upscaled_ch, mamba_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mamba_dim)
        )
        self.mamba_block = Mamba(d_model=mamba_dim, d_state=kwargs.get('ssm_d_state', 1),
                                 d_conv=kwargs.get('ssm_d_conv', 4), expand=kwargs.get('ssm_expand', 2))
        self.gated_residual = GatedResidual(num_channels=mamba_dim)
        self.final_proj = nn.Conv2d(mamba_dim, num_channels, kernel_size=1, bias=False)

    def forward(self, FT1: torch.Tensor, FT2: torch.Tensor) -> torch.Tensor:
        Fcat = concat_features(FT1, FT2)
        Fp = self.pre_mamba_proj(Fcat)
        X_lr, X_rl, X_ud, X_du = four_way_scan(Fp)
        Y_lr, Y_rl, Y_ud, Y_du = mamba_ssm_scan(X_lr, X_rl, X_ud, X_du, self.mamba_block)
        Y_gated_list = self.gated_residual([X_lr, X_rl, X_ud, X_du], [Y_lr, Y_rl, Y_ud, Y_du])
        H, W = FT1.shape[2], FT1.shape[3]
        f_scan = four_way_inverse_scan(Y_gated_list, H, W)
        residual_sum = f_scan + Fp
        output = self.final_proj(residual_sum)
        return output


class S2FMFusion(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, r1: int = 4, r2: int = 8, **kwargs):
        super().__init__()
        self.s2fm = S2FM(num_channels=out_ch, r1=r1, r2=r2, **kwargs)

    def forward(self, f1, f2):
        return self.s2fm(f1, f2)


# =============================================================================
# SECTION 2: DECODER 模块 (来自您的 fullnet.py)
# =============================================================================
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x);
        B, H, W, C = x.shape
        return self.norm(rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4))


class FinalUpsample_X4(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = self.linear1(x).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()
        x = self.linear2(x).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class Mamba_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, drop_path=0., norm_layer=nn.LayerNorm, upsample=None, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([CVSSDecoderBlock(hidden_dim=dim, drop_path=drop_path[i] if isinstance(drop_path,
                                                                                                           list) else drop_path,
                                                      norm_layer=norm_layer, act_layer=nn.GELU, **kwargs) for i in
                                     range(depth)])
        self.upsample = upsample(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer) if upsample else None

    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return self.upsample(x) if self.upsample else x


class MambaDecoder(nn.Module):
    def __init__(self, img_size, in_channels, num_classes, embed_dim, patch_size, depths, drop_path_rate, norm_layer,
                 deep_supervision, **kwargs):
        super().__init__()
        self.num_layers, self.deep_supervision = len(depths), deep_supervision
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers_up = nn.ModuleList()
        for i in range(self.num_layers):
            idx, dim, res = self.num_layers - 1 - i, in_channels[self.num_layers - 1 - i], (
                self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i)),
                self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i)))
            self.layers_up.append(
                PatchExpand(res, dim, norm_layer=norm_layer) if i == 0 else Mamba_up(dim, res, depths[idx], dpr[
                                                                                                            sum(depths[
                                                                                                                :idx]):sum(
                                                                                                                depths[
                                                                                                                :idx + 1])],
                                                                                     norm_layer,
                                                                                     PatchExpand if i < self.num_layers - 1 else None,
                                                                                     **kwargs))
        self.norm_up = norm_layer(embed_dim)
        self.up = FinalUpsample_X4(self.patches_resolution, embed_dim, patch_size, norm_layer)
        self.output = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=False)

    def forward_up_features(self, inputs):
        y, aux = None, []
        for i, layer_up in enumerate(self.layers_up):
            y = layer_up(inputs[i].permute(0, 2, 3, 1).contiguous() if i == 0 else y + inputs[i].permute(0, 2, 3,
                                                                                                         1).contiguous())
            if self.deep_supervision and i < self.num_layers - 1: aux.append(y)
        final = self.norm_up(y)
        return (final, aux) if self.deep_supervision else final

    def forward(self, inputs):
        out = self.forward_up_features(inputs)
        if self.deep_supervision:
            main_out = self.output(self.up(out[0]).permute(0, 3, 1, 2).contiguous())
            return main_out, out[1]
        return self.output(self.up(out).permute(0, 3, 1, 2).contiguous())


# =============================================================================
# SECTION 3: THE BASELINE MODEL (STMbaseS2FM)
# =============================================================================
class STMbaseS2FM(nn.Module):
    def __init__(
            self,
            img_size: list[int] = [256, 256], num_classes: int = 1, in_channels: int = 3,
            dims: list[int] = [64, 128, 256, 512], depths: list[int] = [2, 2, 6, 2],
            drop_path_rate: float = 0.0, deep_supervision: bool = True,
            s2fm_r1: int = 4, s2fm_r2: int = 8,
            **kwargs,
    ):
        super().__init__()
        embed_dim = dims[0]
        self.dims = dims
        self.depths = depths
        self.deep_supervision = deep_supervision

        # --- Encoder setup (与 FullNet 一致, 但使用标准 VSSBlock) ---
        self.stem = nn.Sequential(nn.Conv2d(in_channels, self.dims[0], 4, 4), nn.BatchNorm2d(self.dims[0]))
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        dp_idx = 0
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i in range(4):
            if i > 0: self.downsample_layers.append(
                nn.Sequential(nn.BatchNorm2d(self.dims[i - 1]), nn.Conv2d(self.dims[i - 1], self.dims[i], 2, 2)))
            # [核心差异]: 使用标准的 VSSBlock
            stage = nn.Sequential(*[
                VSSBlock(hidden_dim=self.dims[i], drop_path=dpr[dp_idx + j], norm_layer=LayerNorm2d, channel_first=True,
                         **kwargs) for j in range(depths[i])])
            self.encoder_stages.append(stage)
            dp_idx += depths[i]

        # --- Fusion and Decoder (与 FullNet 完全一致) ---
        self.fusions = nn.ModuleList([S2FMFusion(c, c, r1=s2fm_r1, r2=s2fm_r2, **kwargs) for c in self.dims])
        self.decoder = MambaDecoder(img_size, self.dims, num_classes, embed_dim, 4, depths, drop_path_rate,
                                    nn.LayerNorm, deep_supervision, **kwargs)
        if self.deep_supervision:
            self.deep_supervision_heads = nn.ModuleList(
                [nn.Conv2d(dim, num_classes, 1) for dim in [self.dims[2], self.dims[1], self.dims[0]]])

    def forward(self, t1: torch.Tensor, t2: torch.Tensor):
        H, W = t1.shape[-2:]

        # --- 标准 Encoder 前向传播 ---
        x1, x2 = self.stem(t1), self.stem(t2)
        feats_t1, feats_t2 = [], []
        for i in range(len(self.depths)):
            x1 = self.encoder_stages[i](x1)
            x2 = self.encoder_stages[i](x2)
            feats_t1.append(x1)
            feats_t2.append(x2)
            if i < len(self.depths) - 1:
                x1 = self.downsample_layers[i](x1)
                x2 = self.downsample_layers[i](x2)

        # --- Fusion and Decoder 前向传播 (与 FullNet 完全一致) ---
        skips = [self.fusions[i](feats_t1[i], feats_t2[i]) for i in range(len(self.depths))]
        decoder_output = self.decoder(list(reversed(skips)))

        if self.deep_supervision:
            main_output, aux_features = decoder_output
            outputs = [main_output]
            for i, aux_feat in enumerate(aux_features):
                aux_pred = self.deep_supervision_heads[i](aux_feat.permute(0, 3, 1, 2).contiguous())
                outputs.append(F.interpolate(aux_pred, (H, W), mode='bilinear', align_corners=False))
            return outputs
        else:
            return decoder_output


# =============================================================================
# SECTION 4: SELF-TESTING BLOCK
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}\n")

    # 实例化模型，参数尽量与 FullNet 保持一致以作对比
    model = STMbaseS2FM(
        img_size=[256, 256],
        dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        deep_supervision=True,
        s2fm_r1=4,
        s2fm_r2=4
    ).to(device)

    model.eval()
    t1 = torch.randn(2, 3, 256, 256, device=device)
    t2 = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        outputs = model(t1, t2)

    print("✅ 模型成功执行前向传播。")
    print(f"模型返回 {len(outputs)} 个输出 (1 主 + {len(outputs) - 1} 辅)。")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n模型可训练参数量: {n_params:.2f}M")
    print("✅ 精确基线模型构建完毕。")


