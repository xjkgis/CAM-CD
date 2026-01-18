import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
from functools import partial

try:
    from VMamba.classification.models.csm_triton import cross_scan_fn, cross_merge_fn
except ImportError:
    try:
        from VMamba.classification.models.csm_triton import cross_scan_fn, cross_merge_fn
    except ImportError:
        print("Warning: csm_triton not found. Cross scan functions might fail.")

try:
    from VMamba.classification.models.csms6s import selective_scan_fn
except ImportError:
    try:
        from VMamba.classification.models.csms6s import selective_scan_fn
    except ImportError:
        print("Warning: csms6s not found. Selective scan might fail.")

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in
                    range(k_group)]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class PriorToWeights(nn.Module):
    def __init__(self, in_ch=4, hidden=32, weight_norm="sigmoid"):
        super().__init__()
        self.projs = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, kernel_size=1),
        )
        if weight_norm == "sigmoid":
            self.norm = nn.Sigmoid()
        elif weight_norm == "softmax":
            self.norm = partial(F.softmax, dim=1)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.projs(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SS2D_CAAS(nn.Module):
    def __init__(
            self,
            d_model,
            d_inner,
            d_state=1,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.2,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            channel_first=False,
            weight_hidden=32,
            weight_norm="sigmoid",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.k_group = 4
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1

        Linear = Linear2d if self.channel_first else nn.Linear

        d_proj = self.d_inner * 2
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act = act_layer()

        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )

        x_proj_list = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj_list], dim=0))  # (K, N, inner)
        del x_proj_list

        if initialize == "v0":
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        else:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.k_group, self.d_inner)))

        self.out_norm = LayerNorm2d(self.d_inner) if self.channel_first else nn.LayerNorm(self.d_inner)
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.prior2w = PriorToWeights(in_ch=4, hidden=weight_hidden, weight_norm=weight_norm)

    def forward(self, x: torch.Tensor, prior=None, alpha=None):
        x_xz = self.in_proj(x)
        x_ssm, z = x_xz.chunk(2, dim=(1 if self.channel_first else -1))
        if not self.channel_first:
            x_ssm = x_ssm.permute(0, 3, 1, 2).contiguous()
            z = z.permute(0, 3, 1, 2).contiguous()

        if self.with_dconv:
            x_ssm = self.conv2d(x_ssm)
        x_ssm = self.act(x_ssm)
        B, D, H, W = x_ssm.shape
        L = H * W
        K, R, N = self.k_group, self.dt_rank, self.d_state
        xs = cross_scan_fn(x_ssm, in_channel_first=True, out_channel_first=True)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        xs = xs.contiguous().view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        As = -self.A_logs.to(torch.float).exp()
        Ds = self.Ds.to(torch.float)
        dt_bias = self.dt_projs_bias.view(-1).to(torch.float)
        ys = selective_scan_fn(
            xs.float(), dts.float(), As, Bs.float(), Cs.float(), Ds, dt_bias,
            delta_softplus=True
        ).view(B, K, D, H, W)
        if prior is not None and alpha is not None:
            guidance_weights = self.prior2w(prior)
            injection_strength = torch.sigmoid(alpha)
            Wdir = torch.lerp(torch.ones_like(guidance_weights), guidance_weights, injection_strength)
            Wdir = Wdir.unsqueeze(2)
            ys = ys * Wdir.to(ys.dtype)
        y = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True)
        y = y.view(B, -1, H, W)
        y = self.out_norm(y)
        y = y * self.act(z)
        if not self.channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()

        out = self.dropout(self.out_proj(y))
        return out

class VSSBlock_CAAS(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0.2,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            ssm_d_state: int = 1,
            ssm_ratio=1.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=False,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            weight_hidden=32,
            weight_norm="sigmoid",
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = False

        # 自动检测 channel_first
        if norm_layer in [nn.BatchNorm2d, LayerNorm2d]:
            channel_first = True

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            d_inner = int(ssm_ratio * hidden_dim)

            self.op = SS2D_CAAS(
                d_model=hidden_dim,
                d_inner=d_inner,
                d_state=ssm_d_state,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                channel_first=channel_first,
                weight_hidden=weight_hidden,
                weight_norm=weight_norm
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=channel_first)

    def forward(self, input: torch.Tensor, prior=None, alpha=None):
        x = input
        if self.ssm_branch:
            x = x + self.drop_path(self.op(self.norm(x), prior=prior, alpha=alpha))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
