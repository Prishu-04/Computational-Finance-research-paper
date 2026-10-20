import torch
import torch.nn as nn

from src.models.common import QUANTILES, SESSION_POSITION_IDX

N_FEATURES = 15
CONTEXT_BARS = 9
N_CLASSES = 3
N_QUANTILES = len(QUANTILES)


# ---------------------------------------------------------------- LSTM / GRU
class RecurrentModel(nn.Module):
    def __init__(self, cell="lstm", hidden=64):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(N_FEATURES, hidden, batch_first=True)
        self.reg_head = nn.Linear(hidden, 1)
        self.cls_head = nn.Linear(hidden, N_CLASSES)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return {"reg": self.reg_head(last).squeeze(-1), "cls": self.cls_head(last)}


# --------------------------------------------------------------- Transformer
class TransformerModel(nn.Module):
    def __init__(self, d_model=32, nhead=4, layers=2):
        super().__init__()
        self.input_proj = nn.Linear(N_FEATURES, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, CONTEXT_BARS, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_embed
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return {"reg": self.reg_head(pooled).squeeze(-1), "cls": self.cls_head(pooled)}


# ------------------------------------------------------------------ PatchTST
class PatchTSTModel(nn.Module):
    def __init__(self, patch_len=3, d_model=16, nhead=2):
        super().__init__()
        assert CONTEXT_BARS % patch_len == 0
        self.n_patches = CONTEXT_BARS // patch_len
        self.patch_len = patch_len
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x):
        b = x.shape[0]
        # (batch, time, channels) -> (batch, channels, time) -> patches
        xc = x.transpose(1, 2)  # (b, channels, time)
        patches = xc.reshape(b, N_FEATURES, self.n_patches, self.patch_len)
        tokens = self.patch_embed(patches)  # (b, channels, n_patches, d_model)
        tokens = tokens.reshape(b * N_FEATURES, self.n_patches, -1) + self.pos_embed
        enc = self.encoder(tokens)  # channel-independent: shared weights, no cross-channel mixing
        enc = enc.mean(dim=1).reshape(b, N_FEATURES, -1)
        pooled = enc.mean(dim=1)  # aggregate channels only AFTER independent encoding
        return {"reg": self.reg_head(pooled).squeeze(-1), "cls": self.cls_head(pooled)}


# ---------------------------------------------------------------------- CARD
class CARDModel(nn.Module):
    def __init__(self, patch_len=3, d_model=16, nhead=2):
        super().__init__()
        self.n_patches = CONTEXT_BARS // patch_len
        self.patch_len = patch_len
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=32, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.channel_attn = nn.MultiheadAttention(d_model, num_heads=nhead, batch_first=True)
        self.channel_embed = nn.Parameter(torch.randn(1, N_FEATURES, d_model) * 0.02)
        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x):
        b = x.shape[0]
        xc = x.transpose(1, 2)
        patches = xc.reshape(b, N_FEATURES, self.n_patches, self.patch_len)
        tokens = self.patch_embed(patches)
        tokens = tokens.reshape(b * N_FEATURES, self.n_patches, -1) + self.pos_embed
        enc = self.temporal_encoder(tokens)
        per_channel = enc.mean(dim=1).reshape(b, N_FEATURES, -1) + self.channel_embed
        # channel-aligned attention: channels attend to each other
        cross, _ = self.channel_attn(per_channel, per_channel, per_channel)
        pooled = cross.mean(dim=1)
        return {"reg": self.reg_head(pooled).squeeze(-1), "cls": self.cls_head(pooled)}


# -------------------------------------------------------------------- Moirai
class MoiraiModel(nn.Module):
    def __init__(self, patch_len=3, d_model=24, nhead=4, layers=2,
                 use_financial_encoder=False, use_regime=False, n_regime_buckets=4):
        super().__init__()
        self.n_patches = CONTEXT_BARS // patch_len
        self.patch_len = patch_len
        self.use_financial_encoder = use_financial_encoder
        self.use_regime = use_regime
        self.n_regime_buckets = n_regime_buckets

        if use_financial_encoder:
            # Candidate C: financial feature adapter -- a small encoder that
            # remixes raw OHLCV+indicator features before patching, instead
            # of feeding dozens of raw indicators directly into the backbone.
            self.financial_encoder = nn.Sequential(
                nn.Linear(N_FEATURES, N_FEATURES), nn.GELU(), nn.LayerNorm(N_FEATURES)
            )

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.channel_id_embed = nn.Embedding(N_FEATURES, d_model)  # "any-variate" tagging
        self.time_pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        if use_regime:
            # Candidate A: intraday regime-aware representation -- an extra
            # token summarizing where in the session this window sits.
            self.regime_embed = nn.Embedding(n_regime_buckets, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=48, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.quantile_head = nn.Linear(d_model, N_QUANTILES)
        self.cls_head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x):
        b = x.shape[0]
        feats = self.financial_encoder(x) if self.use_financial_encoder else x
        xc = feats.transpose(1, 2)  # (b, channels, time)
        patches = xc.reshape(b, N_FEATURES, self.n_patches, self.patch_len)
        tokens = self.patch_embed(patches)  # (b, channels, n_patches, d_model)

        channel_ids = torch.arange(N_FEATURES, device=x.device)
        tokens = tokens + self.channel_id_embed(channel_ids).view(1, N_FEATURES, 1, -1)
        tokens = tokens + self.time_pos_embed.unsqueeze(1)
        tokens = tokens.reshape(b, N_FEATURES * self.n_patches, -1)  # ANY-VARIATE: one joint sequence

        if self.use_regime:
            # bucket session_position (last timestep) into n_regime_buckets
            session_pos = x[:, -1, SESSION_POSITION_IDX].clamp(0, 1)
            bucket = (session_pos * self.n_regime_buckets).long().clamp(max=self.n_regime_buckets - 1)
            regime_tok = self.regime_embed(bucket).unsqueeze(1)  # (b, 1, d_model)
            tokens = torch.cat([tokens, regime_tok], dim=1)

        enc = self.encoder(tokens)
        pooled = enc.mean(dim=1)
        quantiles = self.quantile_head(pooled)
        quantiles, _ = torch.sort(quantiles, dim=1)  # enforce monotonic quantiles
        return {"reg": quantiles[:, QUANTILES.index(0.5)], "quantiles": quantiles,
                "cls": self.cls_head(pooled)}


class OptimizedMoiraiModel(MoiraiModel):
    def __init__(self, **kwargs):
        kwargs.setdefault("use_financial_encoder", True)
        kwargs.setdefault("use_regime", True)
        super().__init__(**kwargs)
