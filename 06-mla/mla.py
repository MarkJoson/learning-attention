"""Multi-head Latent Attention (MLA) —— DeepSeek-V2/V3 的 KV cache 杀手锏。

MLA 的核心不是"怎么算注意力"，而是**把 KV 压缩成一个低维 latent**：推理时只缓存这个
latent（外加一个共享的 RoPE key），而不是每个 head 的完整 K/V。于是 KV cache 比 MHA 小一两个
数量级，比 GQA 还狠。

本文件用纯 PyTorch 实现 MLA，提供两条等价的计算路径：

  - `forward_naive`：先把 latent **重建**成完整 K/V，再做标准注意力（清晰、作为 ground truth）；
  - `forward_absorb`：推理用的高效路径，把上投影矩阵 **吸收（absorb）** 进 Q 和 O，
    **直接在 latent 上**算注意力，从不重建 K/V —— 这正是 MLA 省显存又省算力的关键。

两条路径数学上完全等价（absorb 只是把 naive 的矩阵乘法做了代数重排），`test_mla.py` 会验证。

本章还提供第三条路径 `forward_absorb_triton`：latent 注意力改用**提取自 lightllm 的 MLA
prefill triton kernel**（见 `mla_triton.py`），直接在 latent 维度算 absorb 注意力——是真实的
高效实现。（DeepSeek 的 FlashMLA 是 CUDA、vLLM 的 MLA backend 深度耦合，都不易独立精读；
lightllm 的这份最干净。）
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLAConfig:
    d_model: int = 512        # 隐藏维度
    n_heads: int = 8          # 注意力头数
    q_lora_rank: int = 256    # query 压缩秩
    kv_lora_rank: int = 128   # **KV 压缩秩**（推理缓存的 latent 维度，DeepSeek 用 512）
    qk_nope_head_dim: int = 64  # content（非位置）部分的 head 维度
    qk_rope_head_dim: int = 32  # RoPE（位置）部分的 head 维度，DeepSeek 用 64
    v_head_dim: int = 64        # value 的 head 维度
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(self.weight.dtype)


def _rope_cos_sin(positions, dim, base, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]   # (S, dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x, cos, sin):
    """对 (B, S, H, dim) 的最后一维做旋转位置编码（GPT-NeoX 半旋转约定）。"""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MLA(nn.Module):
    """Multi-head Latent Attention。naive 与 absorb 共享同一组权重。"""

    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        H, dm = cfg.n_heads, cfg.d_model
        self.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Query：压缩到 q_lora_rank，再上投影到 H 个 head 的 [nope; rope]
        self.W_DQ = nn.Linear(dm, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.W_UQ = nn.Linear(cfg.q_lora_rank, H * self.qk_head_dim, bias=False)

        # KV：压缩到 kv_lora_rank（latent，缓存它）；另有共享的 RoPE key（也缓存）
        self.W_DKV = nn.Linear(dm, cfg.kv_lora_rank, bias=False)
        self.W_KR = nn.Linear(dm, cfg.qk_rope_head_dim, bias=False)   # 所有 head 共享
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        # 上投影：从 latent 重建每个 head 的 [k_nope; v]
        self.W_UKV = nn.Linear(cfg.kv_lora_rank, H * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)

        self.W_O = nn.Linear(H * cfg.v_head_dim, dm, bias=False)
        self.scale = self.qk_head_dim ** -0.5

    # ---- 共享的前半段：投影 + 压缩 ----
    def _project(self, h, positions):
        B, S, _ = h.shape
        c = self.cfg
        H = c.n_heads

        c_q = self.q_norm(self.W_DQ(h))                                # (B,S,q_lora)
        q = self.W_UQ(c_q).view(B, S, H, self.qk_head_dim)
        q_nope, q_rope = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)

        c_kv = self.kv_norm(self.W_DKV(h))                            # (B,S,kv_lora)  ← 缓存
        k_rope = self.W_KR(h).view(B, S, 1, c.qk_rope_head_dim)        # (B,S,1,rope)   ← 缓存（共享）

        cos, sin = _rope_cos_sin(positions, c.qk_rope_head_dim, c.rope_base, h.device, h.dtype)
        q_rope = _apply_rope(q_rope, cos, sin)
        k_rope = _apply_rope(k_rope, cos, sin)
        return q_nope, q_rope, c_kv, k_rope

    @staticmethod
    def _causal(score):
        S = score.shape[-1]
        mask = torch.ones(S, S, dtype=torch.bool, device=score.device).tril()
        return score.masked_fill(~mask, float("-inf"))

    def forward_naive(self, h, positions):
        """重建完整 K/V 再做标准注意力（ground truth）。"""
        B, S, _ = h.shape
        c = self.cfg
        H = c.n_heads
        q_nope, q_rope, c_kv, k_rope = self._project(h, positions)

        kv = self.W_UKV(c_kv).view(B, S, H, c.qk_nope_head_dim + c.v_head_dim)
        k_nope, v = kv.split([c.qk_nope_head_dim, c.v_head_dim], dim=-1)

        q = torch.cat([q_nope, q_rope], dim=-1)                       # (B,S,H,qk_head_dim)
        k = torch.cat([k_nope, k_rope.expand(-1, -1, H, -1)], dim=-1)

        score = torch.einsum("bshd,bthd->bhst", q.float(), k.float()) * self.scale
        attn = torch.softmax(self._causal(score), dim=-1)
        o = torch.einsum("bhst,bthd->bshd", attn, v.float()).to(h.dtype)
        return self.W_O(o.reshape(B, S, H * c.v_head_dim))

    def forward_absorb(self, h, positions):
        """推理高效路径：吸收上投影，直接在 latent 上算，不重建 K/V。"""
        B, S, _ = h.shape
        c = self.cfg
        H = c.n_heads
        q_nope, q_rope, c_kv, k_rope = self._project(h, positions)

        # 拆出上投影矩阵：W_UKV.weight 形状 (H*(nope+v), kv_lora)
        w = self.W_UKV.weight.view(H, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
        W_UK = w[:, : c.qk_nope_head_dim, :]    # (H, nope, kv_lora)
        W_UV = w[:, c.qk_nope_head_dim:, :]     # (H, v,    kv_lora)

        # 把 W_UK 吸收进 q_nope：把 query 从 content 空间投到 latent 空间
        q_absorb = torch.einsum("bshn,hnl->bshl", q_nope.float(), W_UK.float())  # (B,S,H,kv_lora)

        # 注意力分数 = content（在 latent 上）+ position（RoPE 流）
        s_content = torch.einsum("bshl,btl->bhst", q_absorb, c_kv.float())
        s_rope = torch.einsum("bshr,btr->bhst", q_rope.float(), k_rope.squeeze(2).float())
        score = (s_content + s_rope) * self.scale
        attn = torch.softmax(self._causal(score), dim=-1)

        # 输出：先在 latent 上加权，再用 W_UV 投回 value 空间
        o_latent = torch.einsum("bhst,btl->bshl", attn, c_kv.float())            # (B,S,H,kv_lora)
        o = torch.einsum("bshl,hvl->bshv", o_latent, W_UV.float()).to(h.dtype)   # (B,S,H,v)
        return self.W_O(o.reshape(B, S, H * c.v_head_dim))

    def forward_absorb_triton(self, h, positions):
        """与 forward_absorb 等价，但 latent 注意力改用**提取自 lightllm 的 MLA triton kernel**。

        这是真实的 MLA prefill kernel：把 absorb 后的 query 与 latent 喂进去，直接在 latent 维度
        算注意力（score = q·c_kv + q_rope·k_rope, out = attn·c_kv），全程不重建 K/V。需 fp16/bf16。
        """
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from mla_triton import context_attention_fwd_no_prompt_cache

        B, S, _ = h.shape
        c = self.cfg
        H = c.n_heads
        q_nope, q_rope, c_kv, k_rope = self._project(h, positions)
        w = self.W_UKV.weight.view(H, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
        W_UK = w[:, : c.qk_nope_head_dim, :]
        W_UV = w[:, c.qk_nope_head_dim:, :]
        q_absorb = torch.einsum("bshn,hnl->bshl", q_nope.float(), W_UK.float()).to(h.dtype)

        # pack 成 kernel 的 varlen 格式 (总 token, head, dim)；latent 作为共享单 head 的 K/V
        q_nope_p = q_absorb.reshape(B * S, H, c.kv_lora_rank).contiguous()
        q_rope_p = q_rope.reshape(B * S, H, c.qk_rope_head_dim).contiguous()
        kv_nope_p = c_kv.reshape(B * S, 1, c.kv_lora_rank).contiguous()
        kv_rope_p = k_rope.reshape(B * S, 1, c.qk_rope_head_dim).contiguous()
        o_lat = torch.empty(B * S, H, c.kv_lora_rank, dtype=h.dtype, device=h.device)
        b_start = torch.arange(0, B * S, S, device=h.device, dtype=torch.int32)
        b_seqlen = torch.full((B,), S, device=h.device, dtype=torch.int32)
        context_attention_fwd_no_prompt_cache(
            q_nope_p, q_rope_p, kv_nope_p, kv_rope_p, o_lat, b_start, b_seqlen, S, self.scale)

        o_lat = o_lat.reshape(B, S, H, c.kv_lora_rank)
        o = torch.einsum("bshl,hvl->bshv", o_lat.float(), W_UV.float()).to(h.dtype)
        return self.W_O(o.reshape(B, S, H * c.v_head_dim))

    # ---- KV cache 账：每个 token 需要缓存多少标量 ----
    def kv_cache_per_token(self) -> int:
        """MLA 推理每 token 缓存：latent c_kv + 共享 RoPE key。"""
        return self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim
