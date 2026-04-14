"""
Attentive Neural Process (ANP) surrogate model for MaaS ABM objectives.

Physics-aware multi-task architecture:
    - Pretrains on 13-dim output: 9 physical intermediates + 4 objectives
    - Finetunes with decoder-first freezing + full-context training
    - Supports continual learning via context set expansion

Key components:
    - ConditionEncoder:    96-dim district features -> 16-dim embedding
    - DeterministicEncoder: per-context-point encoding
    - LatentEncoder:       global latent variable z ~ N(mu, sigma^2)
    - CrossAttention:      multi-head attention from targets to context
    - Decoder:             attended_r + z + theta -> (mu, sigma)
    - AttentiveNeuralProcess: full model (~60K params)
    - ANPTrainer:          pretrain/finetune/continual_finetune manager
    - ANPPredictor:        inference interface for surrogate integration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ANP_THETA_DIM, ANP_Y_DIM, ANP_COND_DIM, ANP_COND_EMBED_DIM,
    ANP_ENCODER_HIDDEN, ANP_DECODER_HIDDEN, ANP_LATENT_DIM,
    ANP_ATTENTION_HEADS, ANP_DET_OUTPUT_DIM,
    THETA_LOWER, THETA_UPPER,
    IDX_OBJ_START, N_INTERMEDIATES,
    MULTITASK_W_INTERMEDIATE, MULTITASK_W_OBJECTIVE, MULTITASK_W_KL,
)


# ================================================================== #
#  Data Normalizer                                                     #
# ================================================================== #

class DataNormalizer:
    """Per-dimension standardization (zero-mean, unit-variance).

    Theta is normalized to [0,1] using fixed bounds.
    Y is per-dimension standardized with partial refit support
    for selective dimension updates during finetuning.
    """

    def __init__(self):
        self.theta_lower = np.array(THETA_LOWER, dtype=np.float64)
        self.theta_upper = np.array(THETA_UPPER, dtype=np.float64)
        self.theta_range = self.theta_upper - self.theta_lower
        self.theta_range[self.theta_range < 1e-12] = 1.0
        self.y_mean = None
        self.y_std = None
        self.fitted = False

    def fit(self, theta, y):
        """Compute normalization statistics from data."""
        self.y_mean = y.mean(axis=0).copy()
        self.y_std = y.std(axis=0).copy()
        self.y_std[self.y_std < 1e-12] = 1.0
        self.fitted = True

    def partial_refit(self, y_new, dims):
        """Refit mean/std only for specified dimensions.

        Parameters
        ----------
        y_new : ndarray (M, D)
        dims : slice or array of int
            Which dimensions to refit.
        """
        self.y_mean[dims] = y_new[:, dims].mean(axis=0)
        self.y_std[dims] = y_new[:, dims].std(axis=0)
        self.y_std[self.y_std < 1e-12] = 1.0

    def normalize_theta(self, theta):
        """Normalize theta to [0, 1]."""
        return (theta - self.theta_lower) / self.theta_range

    def normalize_y(self, y):
        """Standardize y per-dimension."""
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y_norm):
        """Reverse standardization."""
        return y_norm * self.y_std + self.y_mean

    def denormalize_y_std(self, sigma_norm):
        """Scale sigma back to original space."""
        return sigma_norm * np.abs(self.y_std)


# ================================================================== #
#  1. Condition Encoder                                                #
# ================================================================== #

class ConditionEncoder(nn.Module):
    """Condition encoder: compress 96-dim district features to embedding."""

    def __init__(self, input_dim=ANP_COND_DIM, hidden_dim=32,
                 output_dim=ANP_COND_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, c):
        return self.net(c)


# ================================================================== #
#  2. Deterministic Encoder                                            #
# ================================================================== #

class DeterministicEncoder(nn.Module):
    """Deterministic encoder: encode each context point (theta, y, c_embed) -> r."""

    def __init__(self, theta_dim=ANP_THETA_DIM, y_dim=ANP_Y_DIM,
                 cond_dim=ANP_COND_EMBED_DIM,
                 hidden_dims=None, output_dim=ANP_DET_OUTPUT_DIM):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = list(ANP_ENCODER_HIDDEN)
        input_dim = theta_dim + y_dim + cond_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, theta, y, c_embed):
        n_context = theta.shape[1]
        c_exp = c_embed.unsqueeze(1).expand(-1, n_context, -1)
        x = torch.cat([theta, y, c_exp], dim=-1)
        return self.net(x)


# ================================================================== #
#  3. Latent Encoder                                                   #
# ================================================================== #

class LatentEncoder(nn.Module):
    """Latent encoder: encode global latent variable z from context set."""

    def __init__(self, theta_dim=ANP_THETA_DIM, y_dim=ANP_Y_DIM,
                 cond_dim=ANP_COND_EMBED_DIM,
                 hidden_dims=None, latent_dim=ANP_LATENT_DIM):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = list(ANP_ENCODER_HIDDEN)
        input_dim = theta_dim + y_dim + cond_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.mean_head = nn.Linear(prev, latent_dim)
        self.logvar_head = nn.Linear(prev, latent_dim)
        self.pre_net = nn.Sequential(*layers)

    def forward(self, theta, y, c_embed):
        n_points = theta.shape[1]
        c_exp = c_embed.unsqueeze(1).expand(-1, n_points, -1)
        x = torch.cat([theta, y, c_exp], dim=-1)
        h = self.pre_net(x)
        h_agg = h.mean(dim=1)
        z_mean = self.mean_head(h_agg)
        z_logvar = self.logvar_head(h_agg)
        return z_mean, z_logvar


# ================================================================== #
#  4. Cross Attention                                                  #
# ================================================================== #

class CrossAttention(nn.Module):
    """Multi-head cross attention from target queries to context keys/values."""

    def __init__(self, dim=ANP_DET_OUTPUT_DIM, n_heads=ANP_ATTENTION_HEADS):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, query, keys, values):
        attended, _ = self.attention(query, keys, values)
        return attended


# ================================================================== #
#  5. Decoder                                                          #
# ================================================================== #

class Decoder(nn.Module):
    """Decoder: maps attended representation + z + theta to 13-dim output.

    Output layout: [9 intermediates, 4 objectives] all as mu + sigma.
    No separate intermediate head — the model directly predicts all 13 dims.
    """

    def __init__(self, attended_dim=ANP_DET_OUTPUT_DIM, latent_dim=ANP_LATENT_DIM,
                 theta_dim=ANP_THETA_DIM, cond_dim=ANP_COND_EMBED_DIM,
                 hidden_dims=None, n_objectives=ANP_Y_DIM):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = list(ANP_DECODER_HIDDEN)
        input_dim = attended_dim + latent_dim + theta_dim + cond_dim

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h

        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, n_objectives)
        self.log_sigma_head = nn.Linear(prev, n_objectives)

        self.theta_embed = nn.Sequential(
            nn.Linear(theta_dim, theta_dim),
            nn.ReLU(),
        )

    def forward(self, attended_r, z, theta, c_embed):
        n_target = attended_r.shape[1]
        z_exp = z.unsqueeze(1).expand(-1, n_target, -1)
        c_exp = c_embed.unsqueeze(1).expand(-1, n_target, -1)
        theta_emb = self.theta_embed(theta)

        x = torch.cat([attended_r, z_exp, theta_emb, c_exp], dim=-1)
        h = self.backbone(x)

        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)

        return mu, log_sigma


# ================================================================== #
#  6. Attentive Neural Process                                         #
# ================================================================== #

class AttentiveNeuralProcess(nn.Module):
    """Complete ANP model for MaaS objective prediction (13-dim output)."""

    def __init__(self, theta_dim=ANP_THETA_DIM, y_dim=ANP_Y_DIM,
                 cond_dim=ANP_COND_DIM, cond_embed_dim=ANP_COND_EMBED_DIM,
                 encoder_hidden=None, decoder_hidden=None,
                 latent_dim=ANP_LATENT_DIM, attention_heads=ANP_ATTENTION_HEADS,
                 det_output_dim=ANP_DET_OUTPUT_DIM):
        super().__init__()
        if encoder_hidden is None:
            encoder_hidden = list(ANP_ENCODER_HIDDEN)
        if decoder_hidden is None:
            decoder_hidden = list(ANP_DECODER_HIDDEN)

        self.condition_encoder = ConditionEncoder(
            input_dim=cond_dim, output_dim=cond_embed_dim)
        self.det_encoder = DeterministicEncoder(
            theta_dim=theta_dim, y_dim=y_dim, cond_dim=cond_embed_dim,
            hidden_dims=encoder_hidden, output_dim=det_output_dim)
        self.latent_encoder = LatentEncoder(
            theta_dim=theta_dim, y_dim=y_dim, cond_dim=cond_embed_dim,
            hidden_dims=encoder_hidden, latent_dim=latent_dim)
        self.cross_attention = CrossAttention(
            dim=det_output_dim, n_heads=attention_heads)
        self.decoder = Decoder(
            attended_dim=det_output_dim, latent_dim=latent_dim,
            theta_dim=theta_dim, cond_dim=cond_embed_dim,
            hidden_dims=decoder_hidden, n_objectives=y_dim)

        self.target_query_encoder = nn.Sequential(
            nn.Linear(theta_dim + cond_embed_dim, encoder_hidden[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden[0], det_output_dim),
        )

    def forward(self, context_theta, context_y, target_theta, condition,
                target_y=None):
        """
        Parameters
        ----------
        context_theta : Tensor (batch, n_context, 17)
        context_y : Tensor (batch, n_context, 13)
        target_theta : Tensor (batch, n_target, 17)
        condition : Tensor (batch, 96)
        target_y : Tensor (batch, n_target, 13), optional

        Returns
        -------
        dict: 'mu' (batch, n_target, 13), 'sigma' (batch, n_target, 13),
              'z_mean', 'z_logvar', 'z_mean_ctx', 'z_logvar_ctx'
        """
        c_embed = self.condition_encoder(condition)

        # Deterministic path
        r_context = self.det_encoder(context_theta, context_y, c_embed)

        # Target queries
        n_target = target_theta.shape[1]
        c_exp = c_embed.unsqueeze(1).expand(-1, n_target, -1)
        query_input = torch.cat([target_theta, c_exp], dim=-1)
        queries = self.target_query_encoder(query_input)

        # Cross attention
        attended_r = self.cross_attention(queries, r_context, r_context)

        # Latent path
        z_mean_ctx, z_logvar_ctx = self.latent_encoder(
            context_theta, context_y, c_embed)

        if target_y is not None:
            all_theta = torch.cat([context_theta, target_theta], dim=1)
            all_y = torch.cat([context_y, target_y], dim=1)
            z_mean, z_logvar = self.latent_encoder(all_theta, all_y, c_embed)
        else:
            z_mean, z_logvar = z_mean_ctx, z_logvar_ctx

        # Sample z
        if self.training:
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            z = z_mean + eps * std
        else:
            z = z_mean_ctx

        # Decode
        mu, log_sigma = self.decoder(attended_r, z, target_theta, c_embed)
        sigma = 0.01 + 0.99 * F.softplus(log_sigma)

        return {
            'mu': mu,
            'sigma': sigma,
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'z_mean_ctx': z_mean_ctx,
            'z_logvar_ctx': z_logvar_ctx,
        }


# ================================================================== #
#  7. ANP Trainer                                                      #
# ================================================================== #

class ANPTrainer:
    """Training manager with multi-task loss, decoder-first finetuning,
    and continual learning support."""

    def __init__(self, model, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.normalizer = DataNormalizer()
        self._pretrain_data = None

    def compute_loss(self, output, target_y, w_inter=None, w_obj=None, w_kl=None):
        """Multi-task NLL + KL loss.

        Applies separate weights to intermediate dims (0:9) and objective dims (9:13).
        """
        if w_inter is None:
            w_inter = MULTITASK_W_INTERMEDIATE
        if w_obj is None:
            w_obj = MULTITASK_W_OBJECTIVE
        if w_kl is None:
            w_kl = MULTITASK_W_KL

        mu = output['mu']
        sigma = output['sigma']

        # Per-dimension NLL
        nll_all = 0.5 * torch.log(sigma ** 2 + 1e-8) + \
                  0.5 * ((target_y - mu) ** 2) / (sigma ** 2 + 1e-8)

        # Split intermediate vs objective
        nll_inter = nll_all[:, :, :IDX_OBJ_START].mean()
        nll_obj = nll_all[:, :, IDX_OBJ_START:].mean()

        # KL divergence
        z_mean = output['z_mean']
        z_logvar = output['z_logvar']
        z_mean_ctx = output['z_mean_ctx']
        z_logvar_ctx = output['z_logvar_ctx']

        kl = 0.5 * (
            z_logvar_ctx - z_logvar
            + (torch.exp(z_logvar) + (z_mean - z_mean_ctx) ** 2)
            / (torch.exp(z_logvar_ctx) + 1e-8)
            - 1.0
        )
        kl = kl.mean()

        return w_inter * nll_inter + w_obj * nll_obj + w_kl * kl

    def _normalize_data(self, theta_np, y_np, fit=False):
        """Normalize theta and y."""
        if fit or not self.normalizer.fitted:
            self.normalizer.fit(theta_np, y_np)
        theta_norm = self.normalizer.normalize_theta(theta_np).astype(np.float32)
        y_norm = self.normalizer.normalize_y(y_np).astype(np.float32)
        return (torch.FloatTensor(theta_norm).to(self.device),
                torch.FloatTensor(y_norm).to(self.device))

    def pretrain(self, analytical_data, condition, epochs=100, batch_size=256):
        """Phase 0: pretrain on analytical data (13-dim with intermediates).

        Uses context/target splitting to teach meta-learning capability.

        Parameters
        ----------
        analytical_data : dict {'theta': (N, 17), 'y': (N, 13)}
        condition : ndarray (96,)
        """
        self._pretrain_data = {
            'theta': analytical_data['theta'].copy(),
            'y': analytical_data['y'].copy(),
        }
        theta_all, y_all = self._normalize_data(
            analytical_data['theta'], analytical_data['y'], fit=True)
        cond = torch.FloatTensor(condition).unsqueeze(0).to(self.device)
        N = theta_all.shape[0]

        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = perm[start:end]
                batch_theta = theta_all[idx]
                batch_y = y_all[idx]

                B = batch_theta.shape[0]
                n_ctx = max(1, B // 3)
                ctx_idx = torch.randperm(B)[:n_ctx]
                tgt_idx = torch.arange(B)

                ctx_theta = batch_theta[ctx_idx].unsqueeze(0)
                ctx_y = batch_y[ctx_idx].unsqueeze(0)
                tgt_theta = batch_theta[tgt_idx].unsqueeze(0)
                tgt_y = batch_y[tgt_idx].unsqueeze(0)

                output = self.model(ctx_theta, ctx_y, tgt_theta, cond,
                                    target_y=tgt_y)
                loss = self.compute_loss(output, tgt_y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  [Pretrain] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def finetune(self, abm_data, condition, epochs=300, lr=None):
        """Phase A: finetune on ABM data with decoder-first + full-context strategy.

        Parameters
        ----------
        abm_data : dict {'theta': (M, 17), 'y': (M, 13)}
        condition : ndarray (96,)
        epochs : int
        lr : float or None (uses config default)
        """
        # Refit normalizer: keep intermediate dims (0:9), refit objective dims (9:13)
        self.normalizer.partial_refit(abm_data['y'], slice(IDX_OBJ_START, None))

        theta_all, y_all = self._normalize_data(
            abm_data['theta'], abm_data['y'], fit=False)
        cond = torch.FloatTensor(condition).unsqueeze(0).to(self.device)
        M = theta_all.shape[0]

        if lr is None:
            from config import ANP_FINETUNE_LR
            lr = ANP_FINETUNE_LR

        # ---- Phase 1: freeze encoders, only train decoder (1/3 epochs) ----
        phase1_epochs = max(1, epochs // 3)
        phase2_epochs = epochs - phase1_epochs

        # Freeze everything except decoder
        for name, p in self.model.named_parameters():
            p.requires_grad = 'decoder' in name
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        self.model.train()
        for epoch in range(phase1_epochs):
            # Full-context training: all M points as both context and target
            ctx_theta = theta_all.unsqueeze(0)
            ctx_y = y_all.unsqueeze(0)
            tgt_theta = theta_all.unsqueeze(0)
            tgt_y = y_all.unsqueeze(0)

            output = self.model(ctx_theta, ctx_y, tgt_theta, cond, target_y=tgt_y)
            # Higher weight on objectives during finetune
            loss = self.compute_loss(output, tgt_y,
                                     w_inter=0.3, w_obj=1.0, w_kl=0.05)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  [Finetune-P1] Epoch {epoch+1}/{phase1_epochs}, "
                      f"Loss: {loss.item():.4f}")

        # ---- Phase 2: unfreeze all, lower lr (2/3 epochs) ----
        for p in self.model.parameters():
            p.requires_grad = True
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr * 0.1)

        for epoch in range(phase2_epochs):
            ctx_theta = theta_all.unsqueeze(0)
            ctx_y = y_all.unsqueeze(0)
            tgt_theta = theta_all.unsqueeze(0)
            tgt_y = y_all.unsqueeze(0)

            output = self.model(ctx_theta, ctx_y, tgt_theta, cond, target_y=tgt_y)
            loss = self.compute_loss(output, tgt_y,
                                     w_inter=0.3, w_obj=1.0, w_kl=0.05)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  [Finetune-P2] Epoch {epoch+1}/{phase2_epochs}, "
                      f"Loss: {loss.item():.4f}")

    def continual_finetune(self, all_data, condition, epochs=30, lr=None):
        """Phase B continual learning: decoder-only, full-context, low lr.

        Parameters
        ----------
        all_data : dict {'theta': (M, 17), 'y': (M, 13)}
        condition : ndarray (96,)
        epochs : int
        lr : float or None
        """
        if lr is None:
            from config import ANP_CONTINUAL_LR
            lr = ANP_CONTINUAL_LR

        # Normalizer is FROZEN after Phase A to avoid distribution drift.
        # Phase A's normalizer already covers the objective space range.

        theta_all, y_all = self._normalize_data(
            all_data['theta'], all_data['y'], fit=False)
        cond = torch.FloatTensor(condition).unsqueeze(0).to(self.device)

        # Only train decoder
        for name, p in self.model.named_parameters():
            p.requires_grad = 'decoder' in name
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            ctx_theta = theta_all.unsqueeze(0)
            ctx_y = y_all.unsqueeze(0)
            tgt_theta = theta_all.unsqueeze(0)
            tgt_y = y_all.unsqueeze(0)

            output = self.model(ctx_theta, ctx_y, tgt_theta, cond, target_y=tgt_y)
            loss = self.compute_loss(output, tgt_y,
                                     w_inter=0.2, w_obj=1.0, w_kl=0.01)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  [Continual] Epoch {epoch+1}/{epochs}, "
                      f"Loss: {loss.item():.4f}")

        # Restore requires_grad
        for p in self.model.parameters():
            p.requires_grad = True

    def save(self, path):
        """Save model state dict and normalizer."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'normalizer': {
                'y_mean': self.normalizer.y_mean,
                'y_std': self.normalizer.y_std,
            } if self.normalizer.fitted else None,
        }, path)

    def load(self, path):
        """Load model state dict and normalizer."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if checkpoint.get('normalizer') is not None:
            self.normalizer.y_mean = checkpoint['normalizer']['y_mean']
            self.normalizer.y_std = checkpoint['normalizer']['y_std']
            self.normalizer.fitted = True


# ================================================================== #
#  8. ANP Predictor                                                    #
# ================================================================== #

class ANPPredictor:
    """Inference interface for the ANP model.

    Handles normalization internally: accepts raw theta/y, returns raw predictions.
    Predictions are 13-dim (9 intermediates + 4 objectives).
    """

    def __init__(self, model, context_theta, context_y, condition,
                 device='cpu', normalizer=None):
        """
        Parameters
        ----------
        model : AttentiveNeuralProcess
        context_theta : ndarray (n_context, 17)
        context_y : ndarray (n_context, 13)
        condition : ndarray (96,)
        device : str
        normalizer : DataNormalizer or None
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.normalizer = normalizer

        self._ctx_theta_list = [np.array(context_theta, dtype=np.float64)]
        self._ctx_y_list = [np.array(context_y, dtype=np.float64)]
        self._condition = np.array(condition, dtype=np.float32)

        self._ctx_theta_tensor = None
        self._ctx_y_tensor = None
        self._cond_tensor = None
        self._rebuild_tensors()

    def _rebuild_tensors(self):
        """Rebuild cached tensors from lists, applying normalization."""
        ctx_theta = np.concatenate(self._ctx_theta_list, axis=0)
        ctx_y = np.concatenate(self._ctx_y_list, axis=0)

        if self.normalizer is not None and self.normalizer.fitted:
            ctx_theta = self.normalizer.normalize_theta(ctx_theta)
            ctx_y = self.normalizer.normalize_y(ctx_y)

        self._ctx_theta_tensor = torch.FloatTensor(
            ctx_theta.astype(np.float32)).unsqueeze(0).to(self.device)
        self._ctx_y_tensor = torch.FloatTensor(
            ctx_y.astype(np.float32)).unsqueeze(0).to(self.device)
        self._cond_tensor = torch.FloatTensor(
            self._condition).unsqueeze(0).to(self.device)

    def predict(self, theta):
        """Predict 13-dim output (raw scale).

        Parameters
        ----------
        theta : ndarray (n, 17) or (17,)

        Returns
        -------
        mu : ndarray (n, 13)
        sigma : ndarray (n, 13)
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))

        if self.normalizer is not None and self.normalizer.fitted:
            theta_norm = self.normalizer.normalize_theta(theta).astype(np.float32)
        else:
            theta_norm = theta.astype(np.float32)

        tgt_theta = torch.FloatTensor(theta_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(
                self._ctx_theta_tensor, self._ctx_y_tensor,
                tgt_theta, self._cond_tensor)

        mu_norm = output['mu'].squeeze(0).cpu().numpy()
        sigma_norm = output['sigma'].squeeze(0).cpu().numpy()

        if self.normalizer is not None and self.normalizer.fitted:
            mu = self.normalizer.denormalize_y(mu_norm)
            sigma = self.normalizer.denormalize_y_std(sigma_norm)
        else:
            mu, sigma = mu_norm, sigma_norm

        return mu.astype(np.float64), sigma.astype(np.float64)

    def predict_objectives(self, theta):
        """Predict only the 4 objective dimensions (indices 9:13).

        Parameters
        ----------
        theta : ndarray (n, 17) or (17,)

        Returns
        -------
        mu_obj : ndarray (n, 4)
        sigma_obj : ndarray (n, 4)
        """
        mu, sigma = self.predict(theta)
        return mu[:, IDX_OBJ_START:], sigma[:, IDX_OBJ_START:]

    def add_context(self, new_theta, new_y):
        """Add new context points (O(1) append). Accepts RAW data.

        Parameters
        ----------
        new_theta : ndarray (m, 17) or (17,)
        new_y : ndarray (m, 13) or (13,)
        """
        new_theta = np.atleast_2d(np.asarray(new_theta, dtype=np.float64))
        new_y = np.atleast_2d(np.asarray(new_y, dtype=np.float64))
        self._ctx_theta_list.append(new_theta)
        self._ctx_y_list.append(new_y)
        self._rebuild_tensors()

    @property
    def context_size(self):
        """Number of context points currently stored."""
        return sum(arr.shape[0] for arr in self._ctx_theta_list)


# ================================================================== #
#  9. Condition Vector Aggregation                                     #
# ================================================================== #

def aggregate_condition_vector(agents):
    """Aggregate agent-level attributes into a 96-dim district condition vector.

    Returns
    -------
    condition : ndarray (96,)
        16 districts x 6 features, normalized to [0, 1].
    """
    district = agents.get('district', np.zeros(len(next(iter(agents.values()))),
                                                dtype=int))
    N = len(district)
    n_districts = 16
    n_features = 6

    condition = np.zeros(n_districts * n_features, dtype=np.float64)

    for idx in range(n_districts):
        d = idx
        mask = district == d
        n_d = mask.sum()
        if n_d == 0:
            continue

        offset = idx * n_features
        condition[offset + 0] = agents.get('income1', np.zeros(N))[mask].mean()
        condition[offset + 1] = agents.get('income2', np.zeros(N))[mask].mean()
        condition[offset + 2] = agents.get('income3', np.zeros(N))[mask].mean()
        condition[offset + 3] = agents.get('have_car', np.zeros(N))[mask].mean()
        pt = (agents.get('week_metro', np.zeros(N))[mask].astype(float) +
              agents.get('week_bus', np.zeros(N))[mask].astype(float))
        condition[offset + 4] = np.clip(pt.mean(), 0, 1)
        condition[offset + 5] = agents.get('week_taxi', np.zeros(N))[mask].mean()

    cond_min = condition.min()
    cond_max = condition.max()
    if cond_max - cond_min > 1e-12:
        condition = (condition - cond_min) / (cond_max - cond_min)

    return condition.astype(np.float64)
