from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


try:
    from models.change_prior import ChangePrior
    from models.VSSBlock_CAAS import VSSBlock_CAAS, LayerNorm2d
    from models.baseS2FM import S2FM
    from models.vmamba import CVSSDecoderBlock

    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warning: Could not import project-specific modules: {e}")
    print("Self-testing block will be skipped.")
    IMPORTS_SUCCESSFUL = False

if IMPORTS_SUCCESSFUL:
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
        def __init__(self, dim, input_resolution, depth, drop_path=0., norm_layer=nn.LayerNorm, upsample=None,
                     **kwargs):
            super().__init__()
            self.blocks = nn.ModuleList([CVSSDecoderBlock(hidden_dim=dim,
                                                          drop_path=drop_path[i] if isinstance(drop_path,
                                                                                               list) else drop_path,
                                                          norm_layer=norm_layer, act_layer=nn.GELU, **kwargs) for i in
                                         range(depth)])
            self.upsample = upsample(input_resolution, dim=dim, dim_scale=2,
                                     norm_layer=norm_layer) if upsample else None

        def forward(self, x):
            for blk in self.blocks: x = blk(x)
            return self.upsample(x) if self.upsample else x


    class MambaDecoder(nn.Module):
        def __init__(self, img_size, in_channels, num_classes, embed_dim, patch_size, depths, drop_path_rate,
                     norm_layer, deep_supervision, **kwargs):
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


    class S2FMFusion(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, r1: int = 4, r2: int = 8, **kwargs):
            super().__init__()
            self.s2fm = S2FM(num_channels=out_ch, r1=r1, r2=r2, **kwargs)

        def forward(self, f1, f2):
            return self.s2fm(f1, f2)

    class EncoderStage_CAAS(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.blocks = nn.ModuleList(blocks)

        def forward(self, x, prior, alpha):  # ← Added alpha parameter
            for blk in self.blocks:
                x = blk(x, prior=prior, alpha=alpha)
            return x

    class FullNet(nn.Module):
        def __init__(
                self,
                img_size: list[int] = [256, 256], num_classes: int = 1, in_channels: int = 3,
                dims: list[int] = [64, 128, 256, 512], depths: list[int] = [2, 2, 6, 2],
                drop_path_rate: float = 0.0, deep_supervision: bool = True,
                s2fm_r1: int = 4, s2fm_r2: int = 8,
                ssm_ratio: float = 2.0,
                alpha_inits: list[float] = [0.0, 0.0, 0.0, 0.0],
                **kwargs,
        ):
            super().__init__()
            embed_dim = dims[0]
            self.dims = dims
            self.deep_supervision = deep_supervision
            self.alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(val, dtype=torch.float32))
                for val in alpha_inits
            ])

            self.stem = nn.Sequential(nn.Conv2d(in_channels, self.dims[0], 4, 4), nn.BatchNorm2d(self.dims[0]))
            dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
            dp_idx = 0
            self.encoder_stages = nn.ModuleList()
            self.downsample_layers = nn.ModuleList()

            for i in range(4):
                if i > 0:
                    self.downsample_layers.append(
                        nn.Sequential(nn.BatchNorm2d(self.dims[i - 1]), nn.Conv2d(self.dims[i - 1], self.dims[i], 2, 2))
                    )

                stage_blocks = [
                    VSSBlock_CAAS(
                        hidden_dim=self.dims[i],
                        drop_path=dpr[dp_idx + j],
                        norm_layer=LayerNorm2d,
                        ssm_ratio=ssm_ratio,
                        **kwargs
                    )
                    for j in range(depths[i])
                ]
                self.encoder_stages.append(EncoderStage_CAAS(stage_blocks))
                dp_idx += depths[i]

            self.fusions = nn.ModuleList([
                S2FMFusion(c, c, r1=s2fm_r1, r2=s2fm_r2, **kwargs)
                for c in self.dims
            ])

            self.prior_generator = ChangePrior(
                in_channels=in_channels,
                num_dirs=4,
                hidden_channels=32,
                num_layers=3,
                downsample=4,
                return_upsampled=False
            )

            decoder_kwargs = kwargs.copy()
            decoder_kwargs['ssm_ratio'] = ssm_ratio
            self.decoder = MambaDecoder(
                img_size, self.dims, num_classes, embed_dim, 4, depths,
                drop_path_rate, nn.LayerNorm, deep_supervision, **decoder_kwargs
            )

            if self.deep_supervision:
                self.deep_supervision_heads = nn.ModuleList([
                    nn.Conv2d(dim, num_classes, 1)
                    for dim in [self.dims[2], self.dims[1], self.dims[0]]
                ])

        def forward(self, t1: torch.Tensor, t2: torch.Tensor):
            H, W = t1.shape[-2:]
            sizes = [(H // (4 * 2 ** i), W // (4 * 2 ** i)) for i in range(4)]
            p_static, _ = self.prior_generator(t1, t2)
            x1, x2 = self.stem(t1), self.stem(t2)
            feats_t1, feats_t2 = [], []
            for i in range(4):
                p_sized = F.interpolate(p_static, size=sizes[i], mode='bilinear', align_corners=False)
                x1 = self.encoder_stages[i](x1, prior=p_sized, alpha=self.alphas[i])
                x2 = self.encoder_stages[i](x2, prior=p_sized, alpha=self.alphas[i])
                feats_t1.append(x1)
                feats_t2.append(x2)
                if i < 3:
                    x1 = self.downsample_layers[i](x1)
                    x2 = self.downsample_layers[i](x2)
            skips = [self.fusions[i](feats_t1[i], feats_t2[i]) for i in range(4)]
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
