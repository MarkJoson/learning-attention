# =============================================================================
# 来源标注 (Provenance) —— 本仓库 06-mla 的 MLA prefill triton kernel
# -----------------------------------------------------------------------------
# 提取自 lightllm 的 deepseek2 MLA kernel，**计算逻辑未改**，仅去除 1 个依赖
# （is_tesla，原用于 BLOCK 选择，改用 torch.cuda.get_device_capability 判断）。
#
#   source repo : https://github.com/ModelTC/lightllm
#   source file : lightllm/models/deepseek2/triton_kernel/context_flashattention_nopad.py
#   commit      : 9ae6a5d4886312bd827295f3cb0de231639f0c77
#   license     : Apache-2.0
#   取用部分    : _fwd_kernel_no_prompt_cache + context_attention_fwd_no_prompt_cache
#                （MLA prefill，无 prefix cache 版）
#
# 接口（absorb 格式，直接在 latent 维度计算）：
#   context_attention_fwd_no_prompt_cache(q_nope, q_rope, kv_nope, kv_rope, o, ...)
#   其中  score = q_nope·kv_nope + q_rope·kv_rope,  out = attn·kv_nope
#   q_nope 是已 absorb 到 latent (kv_lora_rank) 维度的 query，kv_nope 是 latent c_kv。
# 详见同目录 SOURCES.md 与仓库根 NOTICE。
# =============================================================================
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel_no_prompt_cache(
    Q_nope,
    Q_rope,
    KV_nope,
    KV_rope,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,  # B_LOC 内部记录每个batch 输入的真实位置， B_SEQ_len 记录当前输入的真实长度
    Out,
    stride_q_bs,
    stride_q_h,
    stride_q_d,
    stride_q_rope_bs,
    stride_q_rope_h,
    stride_q_rope_d,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    stride_kv_rope_bs,
    stride_kv_rope_h,
    stride_kv_rope_d,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_ROPE_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num
    cur_kv_head = 0

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_rope_d = tl.arange(0, BLOCK_ROPE_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_q_bs
        + cur_head * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    off_rope_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_q_rope_bs
        + cur_head * stride_q_rope_h
        + offs_rope_d[None, :] * stride_q_rope_d
    )
    off_kv = offs_n[None, :] * stride_kv_bs + cur_kv_head * stride_kv_h + offs_d[:, None] * stride_kv_d
    off_rope_kv = (
        offs_n[None, :] * stride_kv_rope_bs + cur_kv_head * stride_kv_rope_h + offs_rope_d[:, None] * stride_kv_rope_d
    )

    q = tl.load(Q_nope + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    q_rope = tl.load(Q_rope + off_rope_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    kv_ptrs = KV_nope + off_kv
    kv_rope_ptrs = KV_rope + off_rope_kv

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        kv = tl.load(
            kv_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kv_bs,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
            other=0.0,
        )
        kv_rope = tl.load(
            kv_rope_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kv_rope_bs,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, kv)
        qk += tl.dot(q_rope, kv_rope)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.trans(kv)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return



def context_attention_fwd_no_prompt_cache(
    q_nope, q_rope, kv_nope, kv_rope, o, b_start_loc, b_seq_len, max_input_len, softmax_scale
):
    q_nope_dim = q_nope.shape[-1]
    q_rope_dim = q_rope.shape[-1]
    assert q_nope_dim == kv_nope.shape[-1]
    assert q_rope_dim == kv_rope.shape[-1]
    assert q_nope_dim in {16, 32, 64, 128, 256, 512}
    assert q_rope_dim in {16, 32, 64, 128, 256}

    # RTX 4090 适配（本仓库微调，不改 kernel 计算逻辑）：原始 BLOCK 为 A100/H100 调优，
    # head 维度较大时在 4090 (sm_89, ~99KB 共享内存) 上会 OutOfResources，这里相应收紧。
    _cap = torch.cuda.get_device_capability()[0]
    if q_nope_dim >= 512:
        BLOCK = 32 if _cap >= 9 else 16
    elif q_nope_dim >= 256:
        BLOCK = 64 if _cap >= 9 else 32
    else:
        BLOCK = 128

    if q_nope.dtype == torch.float32:
        BLOCK = BLOCK // 4

    sm_scale = softmax_scale
    batch, head = b_seq_len.shape[0], q_nope.shape[1]
    kv_group_num = q_nope.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,

    num_warps = 4 if q_nope_dim <= 64 else 8
    _fwd_kernel_no_prompt_cache[grid](
        q_nope,
        q_rope,
        kv_nope,
        kv_rope,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_rope.stride(0),
        q_rope.stride(1),
        q_rope.stride(2),
        kv_nope.stride(0),
        kv_nope.stride(1),
        kv_nope.stride(2),
        kv_rope.stride(0),
        kv_rope.stride(1),
        kv_rope.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=q_nope_dim,
        BLOCK_ROPE_DMODEL=q_rope_dim,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return