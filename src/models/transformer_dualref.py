import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def forward(self, x):
        return gelu(x)


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None, rel_pos=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if rel_pos is not None:
            att = att + rel_pos
        if mask is not None:
            att = att.masked_fill(mask == 1, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class AxialAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, H, W,
                 add_rel_pos=True, rel_pos_bins=32):
        super().__init__()
        self.rln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.cln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-4)
        self.attn_row = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn_col = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )
        self.add_rel_pos = add_rel_pos
        self.rel_pos_bins = rel_pos_bins
        if add_rel_pos:
            self.row_rel_emb = nn.Parameter(torch.zeros(2 * rel_pos_bins - 1, n_head))
            self.col_rel_emb = nn.Parameter(torch.zeros(2 * rel_pos_bins - 1, n_head))
            nn.init.normal_(self.row_rel_emb, std=0.02)
            nn.init.normal_(self.col_rel_emb, std=0.02)
        self.H = H
        self.W = W

    def _build_rel_pos(self, length, rel_emb, device):
        coords = torch.arange(length, device=device)
        rel = coords[None, :] - coords[:, None]
        rel = rel.clamp(-(self.rel_pos_bins - 1), self.rel_pos_bins - 1) + (self.rel_pos_bins - 1)
        bias = rel_emb[rel]  # [L,L,n_head]
        return bias.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,n_head,L,L]

    def forward(self, x):
        b, c, h, w = x.shape
        x0 = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        x_row = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
        row_rel_pos = self._build_rel_pos(w, self.row_rel_emb, x.device) if self.add_rel_pos else None
        if row_rel_pos is not None:
            row_rel_pos = row_rel_pos.repeat(b * h, 1, 1, 1)
        x_row = self.attn_row(self.rln1(x_row), rel_pos=row_rel_pos)
        x_row = x_row.reshape(b, h, w, c).reshape(b, h * w, c)

        x_col = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        col_rel_pos = self._build_rel_pos(h, self.col_rel_emb, x.device) if self.add_rel_pos else None
        if col_rel_pos is not None:
            col_rel_pos = col_rel_pos.repeat(b * w, 1, 1, 1)
        x_col = self.attn_col(self.cln1(x_col), rel_pos=col_rel_pos)
        x_col = x_col.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b, h * w, c)

        x = x0 + x_row + x_col
        x = x + self.ff(self.ln2(x))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x


class BlockAxial(AxialAttention):
    def __init__(self, config):
        super().__init__(config.n_embd, config.n_head, config.attn_pdrop, config.resid_pdrop, 32, 32)
