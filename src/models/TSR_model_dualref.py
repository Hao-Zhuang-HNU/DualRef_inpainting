import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from .transformer_dualref import BlockAxial, GELU

logger = logging.getLogger(__name__)


class EdgeLineGPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    use_ref_kv = True
    block_size = 1024
    global_pool_size = 16
    local_pool_size = 32

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class GlobalCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, ref_feat):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        ref = ref_feat.squeeze(-1).transpose(1, 2)
        q = self.query(x_flat).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(ref).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(ref).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        y = self.resid_drop(self.proj(y))
        y = y.transpose(1, 2).view(B, C, H, W)
        return y


class HybridBlockDualRef(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = BlockAxial(config)
        self.use_ref_kv = getattr(config, 'use_ref_kv', True)
        self.ln_cross = nn.LayerNorm(config.n_embd)
        self.ln_ffn = nn.LayerNorm(config.n_embd)
        if self.use_ref_kv:
            self.cross_attn = GlobalCrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, ref_feat=None):
        x = x + self.self_attn(x)
        if self.use_ref_kv and ref_feat is not None:
            x_norm = self.ln_cross(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + self.cross_attn(x_norm, ref_feat)
        x_ffn = self.mlp(self.ln_ffn(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        x = x + x_ffn
        return x


class EdgeLineGPT256RelDualRef(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=7, padding=0)
        self.act = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 256))
        self.type_emb_global = nn.Parameter(torch.zeros(1, 1, 256))
        self.type_emb_local = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.normal_(self.type_emb_global, std=0.02)
        nn.init.normal_(self.type_emb_local, std=0.02)

        self.ref_refinement = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )

        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([HybridBlockDualRef(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(256)
        self.block_size = 32
        self.config = config
        self.use_ref_kv = getattr(config, 'use_ref_kv', True)

        self.line_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7, padding=0),
        )
        self.act_last = nn.Sigmoid()
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        no_decay.update({'pos_emb', 'type_emb_global', 'type_emb_local'})
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Parameters such as axial relative position embeddings (e.g. row_rel_emb/col_rel_emb)
        # are standalone nn.Parameter tensors (not "*.weight"/"*.bias"), so place them in
        # no_decay by default instead of failing the optimizer split assertion.
        for pn in param_dict.keys():
            if pn not in decay and pn not in no_decay:
                no_decay.add(pn)
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"params in both decay/no_decay: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, f"params not separated: {param_dict.keys() - union_params}"
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

    def _encode(self, img_idx, line_idx, masks):
        img_idx = img_idx * (1 - masks)
        line_idx = line_idx * (1 - masks)
        x = torch.cat((img_idx, line_idx, masks), dim=1)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2).contiguous()
        position_embeddings = self.pos_emb[:, :h * w, :]
        x = self.drop(x + position_embeddings)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x

    def _decode(self, x):
        x = self.ln_f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        line = self.line_decoder(x)
        return line

    def extract_reference_features(self,
                                   global_img=None, global_line=None,
                                   local_img=None, local_line=None, local_mask=None):
        ref_list = []
        if global_img is not None or global_line is not None:
            if global_img is None:
                B, _, H, W = global_line.shape
                global_img = torch.zeros((B, 3, H, W), device=global_line.device, dtype=global_line.dtype)
            zero_mask = torch.zeros_like(global_img[:, :1, :, :])
            g_feat = self._encode(global_img, global_line, masks=zero_mask)
            g_feat = F.adaptive_avg_pool2d(g_feat, (self.config.global_pool_size, self.config.global_pool_size))
            g_feat = g_feat.flatten(2).transpose(1, 2) + self.type_emb_global
            ref_list.append(g_feat)
        if local_img is not None or local_line is not None:
            if local_img is None:
                B, _, H, W = local_line.shape
                local_img = torch.zeros((B, 3, H, W), device=local_line.device, dtype=local_line.dtype)
            if local_mask is None:
                local_mask = torch.zeros_like(local_line)
            l_feat = self._encode(local_img, local_line, masks=local_mask)
            l_feat = self.ref_refinement(l_feat)
            if getattr(self.config, 'local_pool_size', 32) != 32:
                l_feat = F.adaptive_avg_pool2d(l_feat, (self.config.local_pool_size, self.config.local_pool_size))
            l_feat = l_feat.flatten(2).transpose(1, 2) + self.type_emb_local
            ref_list.append(l_feat)
        if not ref_list:
            return None
        final_ref = torch.cat(ref_list, dim=1).permute(0, 2, 1).unsqueeze(-1)
        return final_ref

    def forward_with_logits(self, img_idx, edge_or_line_idx, line_idx=None, masks=None, ref_feat=None):
        if line_idx is None:
            line_idx = edge_or_line_idx
        x = self._encode(img_idx, line_idx, masks)
        for block in self.blocks:
            x = block(x, ref_feat=ref_feat)
        line = self._decode(x)
        return line

    def forward(self, img_idx, edge_or_line_idx, line_idx=None, edge_targets=None, line_targets=None, masks=None,
                global_img=None, global_edge=None, global_line=None,
                local_img=None, local_edge=None, local_line=None, local_mask=None):
        if line_idx is None:
            line_idx = edge_or_line_idx
        if line_targets is None:
            line_targets = line_idx
        ref_feat = None
        if self.use_ref_kv:
            ref_feat = self.extract_reference_features(
                global_img=global_img, global_line=global_line,
                local_img=local_img, local_line=local_line, local_mask=local_mask,
            )
        line = self.forward_with_logits(img_idx, line_idx, masks=masks, ref_feat=ref_feat)
        loss = 0
        if line_targets is not None:
            loss = F.binary_cross_entropy_with_logits(
                line.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                line_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                reduction='none'
            )
            loss = (loss * masks.view(-1, 1)).mean()
        line = self.act_last(line)
        return line, loss
