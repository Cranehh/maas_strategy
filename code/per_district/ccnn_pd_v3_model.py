"""Per-district CC-CNN v3 — MIP-friendly variant of v2.

Architectural diff vs v2 (`ccnn_pd_v2_model.py`):
  - Removed: CrossDistrictAttention (softmax not MIP-friendly)
  - Removed: LayerNorm (sqrt+division not MIP-friendly)
  - Added capacity:
    * theta encoder hidden: 64 → 128
    * decoder hidden: (256, 128) → (512, 256)
  - Decoder dropout: 0.05 (lower than v2's 0.1, since no LN means less stable)

Result: pure ReLU + Linear + Conv1D. MIP encoding straightforward.
Expected accuracy: ~3-5pp worse than v2 due to no spatial attention,
but offset by larger capacity + 1000 ABM training.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================================================================
# Constants
# ===================================================================
CCNN_PD3_K = 6
CCNN_PD3_THETA_DIM = 12
CCNN_PD3_COND_DIM = 96
CCNN_PD3_COND_PD_DIM = 6
CCNN_PD3_COND_GLOBAL_EMBED = 16
CCNN_PD3_DISTRICT_EMBED = 8
CCNN_PD3_THETA_PD_EMBED = 128       # ↑ 64 → 128 (capacity)
CCNN_PD3_DET_OUTPUT = 64
CCNN_PD3_LATENT = 32
CCNN_PD3_Y_EMBED = 32
CCNN_PD3_DECODER_HIDDEN = (512, 256)  # ↑ from (256, 128)


# ===================================================================
# Normalizer (same as v2)
# ===================================================================
class CCNNPDv3Normalizer:
    def __init__(self, theta_lower, theta_upper, D, y_dim, idx_obj_start=9):
        self.theta_lower = np.asarray(theta_lower, dtype=np.float64)
        self.theta_upper = np.asarray(theta_upper, dtype=np.float64)
        self.theta_range = self.theta_upper - self.theta_lower
        self.theta_range[self.theta_range < 1e-12] = 1.0
        self.D = int(D)
        self.y_dim = int(y_dim)
        self.idx_obj_start = int(idx_obj_start)
        self.y_mean = None
        self.y_std = None
        self.fitted = False

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64)
        assert y.shape[1] == self.D and y.shape[2] == self.y_dim
        self.y_mean = y.mean(axis=0).copy()
        self.y_std = y.std(axis=0).copy()
        self.y_std[self.y_std < 1e-12] = 1.0
        self.fitted = True

    def normalize_theta_dyn(self, theta_dyn_pd):
        return (theta_dyn_pd - self.theta_lower) / self.theta_range

    def normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    def denormalize_y_std(self, sigma_norm):
        return sigma_norm * np.abs(self.y_std)


def reshape_condition_to_pd(condition, D):
    """Reshape (B, 96) → (B, D, 6). Pad zeros for d ≥ 16."""
    if condition.dim() == 1:
        condition = condition.unsqueeze(0)
    B = condition.shape[0]
    cond_pd = condition.reshape(B, 16, 6)
    if D > 16:
        pad = torch.zeros(B, D - 16, 6, dtype=cond_pd.dtype, device=cond_pd.device)
        cond_pd = torch.cat([cond_pd, pad], dim=1)
    elif D < 16:
        cond_pd = cond_pd[:, :D]
    return cond_pd


# ===================================================================
# 1. Global condition encoder
# ===================================================================
class GlobalConditionEncoder(nn.Module):
    def __init__(self, input_dim=CCNN_PD3_COND_DIM, hidden_dim=32,
                 output_dim=CCNN_PD3_COND_GLOBAL_EMBED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.ReLU(),
        )

    def forward(self, c):
        return self.net(c)


# ===================================================================
# 2. Per-district shared 1D-CNN encoder (capacity ↑)
# ===================================================================
class ThetaEncoder1DCNN_Shared(nn.Module):
    """Shared 1D-CNN over (K, 12) per district. Bigger than v2 (output 128 vs 64)."""

    def __init__(self, K=CCNN_PD3_K, theta_dim=CCNN_PD3_THETA_DIM,
                 conv_channels=(64, 128, 128), embed_dim=CCNN_PD3_THETA_PD_EMBED):
        super().__init__()
        self.K = K
        c1, c2, c3 = conv_channels
        self.conv1 = nn.Conv1d(theta_dim, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, padding=1)
        self.proj = nn.Linear(c3 * K, embed_dim)

    def forward(self, theta_pd):
        B, M, K, D, C = theta_pd.shape
        x = theta_pd.permute(0, 1, 3, 4, 2).reshape(B * M * D, C, K)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(B * M * D, -1)
        x = F.relu(self.proj(x))
        return x.reshape(B, M, D, -1)  # (B, M, D, embed_dim)


# -------------------------------------------------------------------
# Differentiable IBP through the conv stack (for box-robust training).
# Interval arithmetic: an affine op y = Wx + b on a box [l, u] maps to
#   l' = W⁺l + W⁻u + b ,  u' = W⁺u + W⁻l + b
# (W⁺ = max(W,0), W⁻ = min(W,0)); ReLU on a box [l, u] → [relu(l), relu(u)].
# -------------------------------------------------------------------
def _conv1d_ibp(conv: nn.Conv1d, lo, hi):
    Wp = conv.weight.clamp(min=0.0)
    Wn = conv.weight.clamp(max=0.0)
    b = conv.bias
    pad = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
    lo2 = F.conv1d(lo, Wp, b, padding=pad) + F.conv1d(hi, Wn, None, padding=pad)
    hi2 = F.conv1d(hi, Wp, b, padding=pad) + F.conv1d(lo, Wn, None, padding=pad)
    return lo2, hi2


def _linear_ibp(lin: nn.Linear, lo, hi):
    Wp = lin.weight.clamp(min=0.0)
    Wn = lin.weight.clamp(max=0.0)
    b = lin.bias
    lo2 = lo @ Wp.t() + hi @ Wn.t() + b
    hi2 = hi @ Wp.t() + lo @ Wn.t() + b
    return lo2, hi2


def ibp_conv_stack(theta_encoder: 'ThetaEncoder1DCNN_Shared', x_lo, x_hi):
    """Differentiable IBP through conv1→relu→conv2→relu→conv3→relu→proj→relu.

    Parameters
    ----------
    theta_encoder : ThetaEncoder1DCNN_Shared
    x_lo, x_hi : (B', C=12, K) interval bounds on the (normalized) θ sequence,
        in the SAME layout the encoder feeds conv1 (permute(...,3,4,2)).

    Returns
    -------
    list of (l, u) tensors — one per ReLU pre-activation layer:
        [(l_conv1, u_conv1), (l_conv2, u_conv2), (l_conv3, u_conv3),
         (l_proj, u_proj)].
    Each l/u has shape matching that layer's output (channels×K for conv,
    embed_dim for proj).
    """
    out = []
    l1, u1 = _conv1d_ibp(theta_encoder.conv1, x_lo, x_hi)
    out.append((l1, u1))
    l1r, u1r = l1.clamp(min=0.0), u1.clamp(min=0.0)
    l2, u2 = _conv1d_ibp(theta_encoder.conv2, l1r, u1r)
    out.append((l2, u2))
    l2r, u2r = l2.clamp(min=0.0), u2.clamp(min=0.0)
    l3, u3 = _conv1d_ibp(theta_encoder.conv3, l2r, u2r)
    out.append((l3, u3))
    l3r, u3r = l3.clamp(min=0.0), u3.clamp(min=0.0)
    lf = l3r.reshape(l3r.shape[0], -1)
    uf = u3r.reshape(u3r.shape[0], -1)
    lp, up = _linear_ibp(theta_encoder.proj, lf, uf)
    out.append((lp, up))
    return out


# ===================================================================
# 3. Y embedding (same as v2)
# ===================================================================
class YEmbed(nn.Module):
    def __init__(self, D, y_dim, embed_dim=CCNN_PD3_Y_EMBED):
        super().__init__()
        self.D = D
        self.y_dim = y_dim
        self.embed = nn.Linear(D * y_dim, embed_dim)

    def forward(self, y_pd):
        B, M, D, ydim = y_pd.shape
        x = y_pd.reshape(B, M, D * ydim)
        return F.relu(self.embed(x))


# ===================================================================
# 4. Per-sample aggregation (no cross-district attn this time)
# ===================================================================
class PerSampleAggregator(nn.Module):
    """Aggregate (B, M, D, embed) → (B, M, embed) via mean + Linear."""

    def __init__(self, dim=CCNN_PD3_THETA_PD_EMBED):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        h = x.mean(dim=2)
        return F.relu(self.proj(h))


# ===================================================================
# 5. Deterministic encoder
# ===================================================================
class DeterministicEncoder(nn.Module):
    def __init__(self, theta_dim=CCNN_PD3_THETA_PD_EMBED,
                 y_embed_dim=CCNN_PD3_Y_EMBED,
                 cond_embed_dim=CCNN_PD3_COND_GLOBAL_EMBED,
                 output_dim=CCNN_PD3_DET_OUTPUT):
        super().__init__()
        in_dim = theta_dim + y_embed_dim + cond_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, output_dim), nn.ReLU(),
            nn.Linear(output_dim, output_dim), nn.ReLU(),
        )

    def forward(self, theta_r, y_embed, c_embed):
        M = theta_r.shape[1]
        c_exp = c_embed.unsqueeze(1).expand(-1, M, -1)
        x = torch.cat([theta_r, y_embed, c_exp], dim=-1)
        return self.net(x)


# ===================================================================
# 6. Latent encoder (no LN)
# ===================================================================
class LatentEncoder(nn.Module):
    def __init__(self, r_dim=CCNN_PD3_DET_OUTPUT, latent_dim=CCNN_PD3_LATENT):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(r_dim, r_dim), nn.ReLU())
        self.mean_head = nn.Linear(r_dim, latent_dim)
        self.logvar_head = nn.Linear(r_dim, latent_dim)

    def forward(self, r_context):
        h = self.pre(r_context).mean(dim=1)
        return self.mean_head(h), self.logvar_head(h)


# ===================================================================
# 7. Cross attention (target ← context) — preserved (linearizable at TR center)
# ===================================================================
class CrossAttention(nn.Module):
    def __init__(self, dim=CCNN_PD3_DET_OUTPUT, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, query, keys, values):
        attended, _ = self.attn(query, keys, values)
        return attended


# ===================================================================
# 8. Tied decoder (capacity ↑, no LN)
# ===================================================================
class DistrictTiedDecoder(nn.Module):
    """Tied decoder: input shared-context + per-district features → y per district.

    NO LayerNorm (MIP-friendly). Compensate w/ larger hidden + dropout 0.05.
    """

    def __init__(self, target_feat_dim=CCNN_PD3_THETA_PD_EMBED,
                 district_embed_dim=CCNN_PD3_DISTRICT_EMBED,
                 cond_pd_dim=CCNN_PD3_COND_PD_DIM,
                 attended_dim=CCNN_PD3_DET_OUTPUT,
                 latent_dim=CCNN_PD3_LATENT,
                 hidden_dims=CCNN_PD3_DECODER_HIDDEN,
                 dropout=0.05, y_dim=33):
        super().__init__()
        self.y_dim = y_dim
        in_dim = (target_feat_dim + district_embed_dim + cond_pd_dim
                  + attended_dim + latent_dim)
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, y_dim)
        self.log_sigma_head = nn.Linear(prev, y_dim)

    def forward(self, target_feat_pd, district_embed_table,
                condition_pd, attended_r, z):
        B, N, D, F_t = target_feat_pd.shape
        d_emb = district_embed_table.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        cond_exp = condition_pd.unsqueeze(1).expand(-1, N, -1, -1)
        attended_exp = attended_r.unsqueeze(2).expand(-1, -1, D, -1)
        z_exp = z.unsqueeze(1).unsqueeze(2).expand(-1, N, D, -1)

        x = torch.cat([target_feat_pd, d_emb, cond_exp, attended_exp, z_exp], dim=-1)
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        return mu, log_sigma


# ===================================================================
# 9. Full v3 model (no cross-district attn, no LN)
# ===================================================================
class ContextConditionedCNNPDv3(nn.Module):
    """v3 — MIP-friendly per-district CC-CNN.

    All-ReLU + Linear + Conv1D. No LayerNorm, no cross-district softmax attention.
    Bigger capacity (encoder 64→128, decoder hidden 256→512).
    """

    def __init__(self, D=17, y_dim=33, idx_obj_start=9, K=CCNN_PD3_K,
                 decoder_dropout=0.05):
        super().__init__()
        self.D = D
        self.y_dim = y_dim
        self.idx_obj_start = idx_obj_start
        self.K = K

        self.global_cond_encoder = GlobalConditionEncoder()
        self.theta_encoder = ThetaEncoder1DCNN_Shared(K=K)
        self.per_sample_agg = PerSampleAggregator()
        self.y_embed = YEmbed(D=D, y_dim=y_dim)
        self.det_encoder = DeterministicEncoder()
        self.latent_encoder = LatentEncoder()
        self.cross_attention = CrossAttention()
        self.target_query_encoder = nn.Sequential(
            nn.Linear(CCNN_PD3_THETA_PD_EMBED + CCNN_PD3_COND_GLOBAL_EMBED,
                      CCNN_PD3_DET_OUTPUT),
            nn.ReLU(),
        )
        self.district_embed = nn.Embedding(D, CCNN_PD3_DISTRICT_EMBED)
        self.decoder = DistrictTiedDecoder(dropout=decoder_dropout, y_dim=y_dim)

    def _encode_pd(self, theta_pd):
        per_d = self.theta_encoder(theta_pd)
        # NOTE: NO CrossDistrictAttention here (MIP-friendly)
        per_sample = self.per_sample_agg(per_d)
        return per_d, per_sample

    def forward(self, context_theta_pd, context_y_pd,
                target_theta_pd, condition, target_y_pd=None):
        c_embed_global = self.global_cond_encoder(condition)
        condition_pd = reshape_condition_to_pd(condition, self.D)

        ctx_per_d, ctx_per_sample = self._encode_pd(context_theta_pd)
        tgt_per_d, tgt_per_sample = self._encode_pd(target_theta_pd)

        ctx_y_emb = self.y_embed(context_y_pd)
        r_context = self.det_encoder(ctx_per_sample, ctx_y_emb, c_embed_global)

        N = target_theta_pd.shape[1]
        c_exp_n = c_embed_global.unsqueeze(1).expand(-1, N, -1)
        query_input = torch.cat([tgt_per_sample, c_exp_n], dim=-1)
        queries = self.target_query_encoder(query_input)
        attended_r = self.cross_attention(queries, r_context, r_context)

        z_mean_ctx, z_logvar_ctx = self.latent_encoder(r_context)
        if target_y_pd is not None:
            tgt_y_emb = self.y_embed(target_y_pd)
            r_target = self.det_encoder(tgt_per_sample, tgt_y_emb, c_embed_global)
            r_all = torch.cat([r_context, r_target], dim=1)
            z_mean, z_logvar = self.latent_encoder(r_all)
        else:
            z_mean, z_logvar = z_mean_ctx, z_logvar_ctx

        if self.training:
            std = torch.exp(0.5 * z_logvar)
            z = z_mean + torch.randn_like(std) * std
        else:
            z = z_mean_ctx

        mu, log_sigma = self.decoder(
            tgt_per_d, self.district_embed.weight, condition_pd, attended_r, z)
        sigma = 0.01 + 0.99 * F.softplus(log_sigma)

        return {
            'mu': mu, 'sigma': sigma,
            'z_mean': z_mean, 'z_logvar': z_logvar,
            'z_mean_ctx': z_mean_ctx, 'z_logvar_ctx': z_logvar_ctx,
        }

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ===================================================================
# 10. Trainer (cosine LR + AdamW, decoder-first finetune)
# ===================================================================
# Names of MIP-encoded ReLU layers in the target path. Hooks on these
# Linear/Conv1d modules capture pre-activations for stability regularization.
# (Context-only modules are excluded — MIP linearizes them at TR center.)
MIP_RELU_PRE_MODULES = (
    'theta_encoder.conv1',
    'theta_encoder.conv2',
    'theta_encoder.conv3',
    'theta_encoder.proj',
    'per_sample_agg.proj',
    'target_query_encoder.0',
    'decoder.backbone.0',
    'decoder.backbone.3',
)


class CCNNPDv3Trainer:
    def __init__(self, model, normalizer, lr=1e-3, device='cpu', weight_decay=0.0,
                 l1_lambda=0.0, stability_lambda=0.0, stability_scale=0.5,
                 box_stability_lambda=0.0, box_delta=0.1):
        """
        Parameters
        ----------
        l1_lambda : float, default 0.0
            Coefficient on sum(|W|) over conv1d/Linear weights in the MIP
            target path. Encourages weight sparsity → fewer mixed binaries
            (Tjeng/Xiao/Tedrake ICLR 2019 Appendix H: 14× MIP speedup with
            90% L1-zeroed weights).
        stability_lambda : float, default 0.0
            Coefficient on mean(exp(-|x| / stability_scale)) over ReLU
            pre-activations in the MIP target path. Penalizes pre-activations
            near 0 (the POINT VALUE) — a weak proxy for one-sidedness.
        stability_scale : float, default 0.5
            Scale of the exp(-|x|/scale) kernel.
        box_stability_lambda : float, default 0.0
            Coefficient on the CERTIFIED box-stability penalty
            mean(relu(-l)·relu(u)) over the conv-stack ReLU pre-activations,
            where (l, u) are IBP-propagated bounds over a box
            [θ_n − box_delta, θ_n + box_delta] around each training θ. This is
            the proper "robust training" that Tjeng's networks have: it forces
            the pre-activation INTERVAL (not just the point) to one side of 0,
            which directly reduces mixed binaries in the MIP. Targets the conv
            stack (conv1/conv2/conv3/proj) where ~85% of binaries live.
        box_delta : float, default 0.1
            Half-width of the box (in normalized [0,1] θ space). 0.1 = a
            20%-wide box, matching the MIP encoding TR width.
        """
        self.model = model.to(device)
        self.normalizer = normalizer
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_lambda = float(l1_lambda)
        self.stability_lambda = float(stability_lambda)
        self.stability_scale = float(stability_scale)
        self.box_stability_lambda = float(box_stability_lambda)
        self.box_delta = float(box_delta)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        # Hooks on MIP-relevant Linear/Conv1d modules (only when enabled)
        self._preact_buffer = {}   # name → tensor (most recent forward call)
        self._hooks = []
        if self.stability_lambda > 0:
            self._register_preact_hooks()

    def _register_preact_hooks(self):
        """Attach forward hooks on MIP-encoded pre-ReLU modules to capture
        their outputs (= pre-activation values). The decoder backbone /
        theta encoder execute multiple times per forward (per zone, per
        ctx/tgt) — the hook overwrites the buffer each call, so we end up
        with the most-recent call's output, which is the target zone's
        pre-activation (what the MIP actually encodes)."""
        named = dict(self.model.named_modules())
        for name in MIP_RELU_PRE_MODULES:
            mod = named.get(name)
            if mod is None:
                # Layer not present (e.g., decoder with different hidden config)
                continue
            def make_hook(layer_name):
                def hook(module, inp, output):
                    # Store grad-attached tensor so the penalty backprops.
                    self._preact_buffer[layer_name] = output
                return hook
            self._hooks.append(mod.register_forward_hook(make_hook(name)))

    def remove_hooks(self):
        """Remove all registered hooks (call when done with trainer)."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._preact_buffer.clear()

    @staticmethod
    def _cosine_lr(epoch, total_epochs, base_lr, warmup_epochs=5, min_lr_ratio=0.05):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cos_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos_factor)

    def _box_stability_penalty(self, target_theta_n):
        """Certified IBP box-stability penalty over the conv stack.

        target_theta_n : (B, M, K, D, 12) normalized θ in [0,1].

        For each conv-stack ReLU pre-activation with IBP bounds (l, u) over the
        box [θ_n − δ, θ_n + δ], the unit is "mixed" iff l < 0 < u. We penalize
        the *normalized stability margin*:
            m = min(relu(-l), relu(u)) / (u - l + eps)   ∈ [0, 0.5]
        — 0 if the unit is stable (one of l≥0 or u≤0), 0.5 if the interval is
        symmetric around 0. Dimensionless ⇒ well-calibrated regardless of
        layer scale. Returns the mean over all units of all 4 conv-stack ReLU
        layers (conv1, conv2, conv3, proj).
        """
        lo = (target_theta_n - self.box_delta).clamp(0.0, 1.0)
        hi = (target_theta_n + self.box_delta).clamp(0.0, 1.0)
        B, M, K, D, C = lo.shape
        # Match ThetaEncoder1DCNN_Shared.forward layout: permute(0,1,3,4,2) → (B,M,D,C,K) → reshape (B*M*D, C, K)
        x_lo = lo.permute(0, 1, 3, 4, 2).reshape(B * M * D, C, K)
        x_hi = hi.permute(0, 1, 3, 4, 2).reshape(B * M * D, C, K)
        ibp_layers = ibp_conv_stack(self.model.theta_encoder, x_lo, x_hi)
        terms = []
        for (l, u) in ibp_layers:
            margin = torch.minimum(F.relu(-l), F.relu(u))      # > 0 only if mixed
            width = (u - l).clamp(min=1e-6)
            terms.append((margin / width).mean())
        return torch.stack(terms).mean()

    def compute_loss(self, output, target_y_norm,
                     w_inter=1.0, w_obj=0.5, w_kl=0.1,
                     target_theta_n=None):
        mu = output['mu']
        sigma = output['sigma']
        nll = (0.5 * torch.log(sigma ** 2 + 1e-8)
               + 0.5 * ((target_y_norm - mu) ** 2) / (sigma ** 2 + 1e-8))
        split = self.model.idx_obj_start
        nll_inter = nll[..., :split].mean()
        if isinstance(w_obj, (torch.Tensor, np.ndarray)):
            if isinstance(w_obj, np.ndarray):
                w_obj_t = torch.as_tensor(w_obj, dtype=mu.dtype, device=mu.device)
            else:
                w_obj_t = w_obj.to(dtype=mu.dtype, device=mu.device)
            obj_term = (nll[..., split:] * w_obj_t).mean()
        else:
            obj_term = w_obj * nll[..., split:].mean()
        z_mean = output['z_mean']
        z_logvar = output['z_logvar']
        z_mean_ctx = output['z_mean_ctx']
        z_logvar_ctx = output['z_logvar_ctx']
        kl = 0.5 * (
            z_logvar_ctx - z_logvar
            + (torch.exp(z_logvar) + (z_mean - z_mean_ctx) ** 2)
            / (torch.exp(z_logvar_ctx) + 1e-8)
            - 1.0
        ).mean()
        total = w_inter * nll_inter + obj_term + w_kl * kl

        # --- Stability regularization (Tjeng/Xiao/Tedrake style) ---
        # Penalize ReLU pre-activations near 0; encourages one-sided ReLUs
        # across the TR so the MIP encoding has fewer mixed binaries.
        if self.stability_lambda > 0 and self._preact_buffer:
            stab_terms = [
                torch.exp(-torch.abs(pre) / self.stability_scale).mean()
                for pre in self._preact_buffer.values()
            ]
            stab = torch.stack(stab_terms).mean()
            total = total + self.stability_lambda * stab
            self._preact_buffer.clear()  # reset for next forward

        # --- L1 sparsity on MIP-target-path weights ---
        # Per Tjeng Appendix H: zeroing small weights post-hoc gives 14× MIP
        # speedup. We apply soft L1 during training to encourage that.
        if self.l1_lambda > 0:
            l1_prefixes = ('theta_encoder.conv', 'theta_encoder.proj',
                           'decoder.backbone', 'per_sample_agg.proj',
                           'target_query_encoder.0')
            l1_terms = [
                p.abs().sum() for n, p in self.model.named_parameters()
                if 'weight' in n and any(p_pref in n for p_pref in l1_prefixes)
            ]
            if l1_terms:
                l1 = torch.stack(l1_terms).sum()
                total = total + self.l1_lambda * l1

        # --- Certified box-stability (the proper Tjeng robust-training term) ---
        if self.box_stability_lambda > 0 and target_theta_n is not None:
            box_pen = self._box_stability_penalty(target_theta_n)
            total = total + self.box_stability_lambda * box_pen

        return total

    def _to_tensor(self, theta_pd_np, y_pd_np):
        theta_n = self.normalizer.normalize_theta_dyn(theta_pd_np).astype(np.float32)
        y_n = self.normalizer.normalize_y(y_pd_np).astype(np.float32)
        return (torch.from_numpy(theta_n).float().to(self.device),
                torch.from_numpy(y_n).float().to(self.device))

    def pretrain(self, theta_pd_np, y_pd_np, condition_np,
                 epochs=200, batch_size=64, verbose=True,
                 w_inter=1.0, w_obj=0.5, w_kl=0.1):
        theta_all, y_all = self._to_tensor(theta_pd_np, y_pd_np)
        cond = (torch.from_numpy(condition_np.astype(np.float32))
                .unsqueeze(0).to(self.device))
        Nall = theta_all.shape[0]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.train()
        history = []
        for epoch in range(epochs):
            lr = self._cosine_lr(epoch, epochs, self.lr, warmup_epochs=5)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            perm = torch.randperm(Nall)
            epoch_loss, n_b = 0.0, 0
            for start in range(0, Nall, batch_size):
                idx = perm[start:start + batch_size]
                bt = theta_all[idx]
                by = y_all[idx]
                B = bt.shape[0]
                n_ctx = max(1, B // 3)
                ctx_idx = torch.randperm(B)[:n_ctx]
                tgt_idx = torch.arange(B)
                ctx_theta = bt[ctx_idx].unsqueeze(0)
                ctx_y = by[ctx_idx].unsqueeze(0)
                tgt_theta = bt[tgt_idx].unsqueeze(0)
                tgt_y = by[tgt_idx].unsqueeze(0)
                out = self.model(ctx_theta, ctx_y, tgt_theta, cond, target_y_pd=tgt_y)
                loss = self.compute_loss(out, tgt_y, w_inter=w_inter, w_obj=w_obj, w_kl=w_kl,
                                          target_theta_n=tgt_theta)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_b += 1
            avg = epoch_loss / max(n_b, 1)
            history.append(avg)
            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                print(f"  [Pretrain] Epoch {epoch+1}/{epochs}, Loss {avg:.4f}, LR {lr:.5f}")
        return history

    def finetune(self, theta_pd_np, y_pd_np, condition_np,
                 epochs=800, decoder_first_epochs=200,
                 patience=200, verbose=True,
                 w_inter=0.3, w_obj=1.0, w_kl=0.05):
        theta_all, y_all = self._to_tensor(theta_pd_np, y_pd_np)
        cond = (torch.from_numpy(condition_np.astype(np.float32))
                .unsqueeze(0).to(self.device))

        def _set_trainable(names_kept):
            for name, p in self.model.named_parameters():
                p.requires_grad = any(k in name for k in names_kept) if names_kept else True

        _set_trainable(('decoder', 'cross_attention'))
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr, weight_decay=self.weight_decay)

        best_loss = float('inf')
        no_improve = 0
        history = []
        self.model.train()

        for epoch in range(epochs):
            base_lr = self.lr if epoch < decoder_first_epochs else self.lr * 0.1
            lr = self._cosine_lr(
                epoch if epoch < decoder_first_epochs else epoch - decoder_first_epochs,
                decoder_first_epochs if epoch < decoder_first_epochs
                else epochs - decoder_first_epochs,
                base_lr, warmup_epochs=5)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            ctx_theta = theta_all.unsqueeze(0)
            ctx_y = y_all.unsqueeze(0)
            tgt_theta = theta_all.unsqueeze(0)
            tgt_y = y_all.unsqueeze(0)

            out = self.model(ctx_theta, ctx_y, tgt_theta, cond, target_y_pd=tgt_y)
            loss = self.compute_loss(out, tgt_y, w_inter=w_inter, w_obj=w_obj, w_kl=w_kl,
                                      target_theta_n=tgt_theta)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            history.append(loss.item())
            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                phase = 'P1-dec' if epoch < decoder_first_epochs else 'P2-full'
                print(f"  [Finetune-{phase}] Epoch {epoch+1}/{epochs}, "
                      f"Loss: {loss.item():.4f}, LR {lr:.5f}")

            if epoch + 1 == decoder_first_epochs:
                _set_trainable(())
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.lr * 0.1, weight_decay=self.weight_decay)
                best_loss = float('inf')
                no_improve = 0

            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"  [Finetune] Early stop at epoch {epoch+1}")
                    break
        return history

    def save(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'y_mean': self.normalizer.y_mean,
            'y_std': self.normalizer.y_std,
            'theta_lower': self.normalizer.theta_lower,
            'theta_upper': self.normalizer.theta_upper,
            'D': self.normalizer.D,
            'y_dim': self.normalizer.y_dim,
            'idx_obj_start': self.normalizer.idx_obj_start,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        self.normalizer.y_mean = ckpt['y_mean']
        self.normalizer.y_std = ckpt['y_std']
        self.normalizer.theta_lower = ckpt['theta_lower']
        self.normalizer.theta_upper = ckpt['theta_upper']
        self.normalizer.theta_range = (
            self.normalizer.theta_upper - self.normalizer.theta_lower)
        self.normalizer.theta_range[self.normalizer.theta_range < 1e-12] = 1.0
        self.normalizer.fitted = True


# ===================================================================
# 11. Predictor (same shape as v2)
# ===================================================================
class CCNNPDv3Predictor:
    def __init__(self, model, normalizer, ctx_theta_pd, ctx_y_pd, condition,
                 device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.normalizer = normalizer
        self.device = device
        self._ctx_theta = np.asarray(ctx_theta_pd, dtype=np.float64)
        self._ctx_y = np.asarray(ctx_y_pd, dtype=np.float64)
        self._condition = np.asarray(condition, dtype=np.float32)
        ctx_t_n = self.normalizer.normalize_theta_dyn(self._ctx_theta).astype(np.float32)
        ctx_y_n = self.normalizer.normalize_y(self._ctx_y).astype(np.float32)
        self._ctx_t = torch.from_numpy(ctx_t_n).float().unsqueeze(0).to(self.device)
        self._ctx_y_t = torch.from_numpy(ctx_y_n).float().unsqueeze(0).to(self.device)
        self._cond_t = torch.from_numpy(self._condition).float().unsqueeze(0).to(self.device)

    def predict(self, theta_pd):
        theta_pd = np.asarray(theta_pd, dtype=np.float64)
        if theta_pd.ndim == 3:
            theta_pd = theta_pd[np.newaxis, :, :, :]
        theta_n = self.normalizer.normalize_theta_dyn(theta_pd).astype(np.float32)
        tgt = torch.from_numpy(theta_n).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(self._ctx_t, self._ctx_y_t, tgt, self._cond_t)
        mu_n = out['mu'].squeeze(0).cpu().numpy()
        sigma_n = out['sigma'].squeeze(0).cpu().numpy()
        mu = self.normalizer.denormalize_y(mu_n)
        sigma = self.normalizer.denormalize_y_std(sigma_n)
        return mu.astype(np.float64), sigma.astype(np.float64)


# ===================================================================
# Self-test
# ===================================================================
def self_test():
    D = 17
    K = 6
    y_dim = 33
    B, M, N = 1, 5, 3

    print("[ccnn_pd_v3 self-test] Building model...")
    model = ContextConditionedCNNPDv3(D=D, y_dim=y_dim, K=K, decoder_dropout=0.05)
    print(f"  Model parameters: {model.num_parameters():,}")

    ctx_theta = torch.randn(B, M, K, D, 12)
    ctx_y = torch.randn(B, M, D, y_dim)
    tgt_theta = torch.randn(B, N, K, D, 12)
    cond = torch.randn(B, 96)

    out = model(ctx_theta, ctx_y, tgt_theta, cond,
                target_y_pd=torch.randn(B, N, D, y_dim))
    print(f"  mu shape:    {out['mu'].shape}")
    assert out['mu'].shape == (B, N, D, y_dim)

    # MIP-friendliness check (no LayerNorm, no MultiheadAttention except cross-attn)
    has_ln = any(isinstance(m, nn.LayerNorm) for m in model.modules())
    print(f"  LayerNorm present: {has_ln} (expect False)")
    assert not has_ln

    print("✓ v3 self-test passed (MIP-friendly)")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
