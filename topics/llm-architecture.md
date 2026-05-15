---
title: LLM 架构深度解析
date: 2026-05-15
categories: [topics]
tags: [LLM, Transformer, 架构, 注意力机制]
---

# LLM 架构深度解析

**采集时间**: 2026-05-15 · **状态**: 持续更新

---

## 概述

大语言模型（LLM）的核心架构基于 Transformer 的 Decoder-only 设计。自 GPT 系列以来，这一范式主导了大模型领域的发展，但在具体实现上已衍生出大量有意义的架构创新。本文从底层组件出发，系统梳理 LLM 架构的设计空间与演进方向。

---

## 一、注意力机制

### 1.1 标准多头注意力

标准注意力计算：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

每个 token 在计算 attention 时需读取所有前驱 token 的 KV 值，这带来了两个根本性挑战：
- **二次复杂度**：计算量随序列长度平方增长
- **KV 缓存膨胀**：推理时需缓存所有历史 token 的 KV 向量

### 1.2 Grouped-Query Attention (GQA)

GQA 是 MHA 与 MQA（Multi-Query Attention）的折中方案。它将 Query 头分组，每组共享一个 Key/Value 头：

| 方案 | KV 头数 | 缓存大小 | 质量影响 |
|------|---------|---------|---------|
| MHA | 等于 Query 头数（如 32） | 大 | 基线 |
| GQA | Query 头数 / 分组数（如 8） | 中 | < 0.5% |
| MQA | 1 | 极小 | 1-2% 下降 |

LLaMA-2 70B 和 LLaMA-3 均采用 GQA（8 组），在推理效率与模型质量之间取得了良好平衡。

### 1.3 Flash Attention

Flash Attention 的核心洞察是：**注意力计算的瓶颈在 HBM 带宽而非算力**。通过分块计算和重排，将计算过程中间结果保留在 SRAM 中，显著减少 HBM 读写：

```
标准实现:  QK^T → HBM → softmax → HBM → PV → HBM     (多次 HBM 读写)
Flash:     QK^T → softmax → PV  全程在 SRAM 中完成    (少量 HBM 读写)
```

实测效果：在 8K 序列长度下，Flash Attention 相比标准实现**显存占用降低 60%**，训练速度提升 **2-4 倍**。

---

## 二、前馈网络

### 2.1 激活函数演进

激活函数的演进路径可以概括为寻求更好的梯度流和训练稳定性：

```
ReLU ──▶ GELU ──▶ SwiGLU
```

SwiGLU 已成为当前主流选择。它本质上是 Swish 激活与门控机制的融合：

```python
def swiglu(x, W, V):
    # x: 输入, W, V: 权重矩阵
    gate = F.silu(torch.mm(x, W))  # 门控路径
    value = torch.mm(x, V)          # 值路径
    return gate * value             # 逐元素相乘
```

相比 ReLU，SwiGLU 在多个下游任务上提升约 **2-3%**，代价是 FFN 参数量增加约 1/3（需维护两组权重）。

### 2.2 FFN 的变体

- **SwiGLU**：LLaMA、Mistral、DeepSeek 系列采用
- **GeGLU**：PaLM 采用
- **GLU 变体的共同特征**：均引入门控机制，输出为 gate(x) × value(x)

---

## 三、归一化与位置编码

### 3.1 Pre-Norm vs Post-Norm

| 方案 | 做法 | 稳定性 | 代表模型 |
|------|------|-------|---------|
| Post-Norm | 残差连接后归一化 | 不稳定，需 warmup | 原始 Transformer |
| Pre-Norm | 子层前归一化 | 稳定，无需 warmup | GPT-3, LLaMA, 几乎所有现代模型 |
| Sandwich-Norm | 子层前后均归一化 | 最稳定 | 少量实验性工作 |

Pre-Norm 已是行业标准，其公式为：

```
output = x + Sublayer(Norm(x))
```

### 3.2 RoPE（旋转位置编码）

RoPE 是目前最广泛使用的位置编码方案。其核心思想是将位置信息编码为旋转矩阵，作用于 Query 和 Key 向量：

- **相对位置感知**：内积结果仅依赖于 token 间的相对位置
- **长度外推**：支持在推理时使用比训练时更长的序列
- **衰减特性**：长距离 token 的 attention 分数自然衰减

RoPE 的扩展方法（用于超长上下文）：
1. **线性插值**：将位置索引缩放，使位置编码适应更长的序列
2. **NTK-aware 插值**：混合高频和低频分量，保留高频细节
3. **YaRN**：结合 NTK 和注意力温度缩放，在 128K 上下文上表现优异

---

## 四、MoE（混合专家）

MoE 是当前扩展模型容量的主流技术。核心思想是通过稀疏激活，以较低的计算成本换取更大的模型容量。

### 4.1 路由机制

```python
def moe_forward(x, experts, router, top_k=2):
    # x: (batch, seq, dim)
    routing_logits = router(x)                  # (b, s, num_experts)
    routing_weights = F.softmax(routing_logits, dim=-1)
    
    # 选出 top-k 专家
    top_k_weights, top_k_indices = torch.topk(
        routing_weights, k=top_k, dim=-1
    )
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    
    output = torch.zeros_like(x)
    for expert_idx, expert in enumerate(experts):
        mask = (top_k_indices == expert_idx).any(dim=-1)
        if mask.any():
            output[mask] += top_k_weights[mask] * expert(x[mask])
    
    return output
```

### 4.2 关键设计决策

- **Top-k 选择**：k=2 是主流选择，部分工作尝试 k=1（推理更快但质量略降）
- **负载均衡**：通过 auxiliary loss 促使路由公平分配 token
- **专家容量**：每个专家的最大 token 处理量，超出部分丢弃或重新路由
- **细粒度专家**：DeepSeek-V3 将专家规模缩小到 1 个 FFN 的粒度，提高路由精度

### 4.3 代表模型对比

| 模型 | 总参数量 | 激活参数 | 专家数 | Top-k |
|------|---------|---------|--------|-------|
| Mixtral 8×7B | 47B | 13B | 8 | 2 |
| DeepSeek-V2 | 236B | 21B | 160 | 6 |
| DeepSeek-V3 | 671B | 37B | 256 | 8 |
| Qwen2.5-MoE | 42B | 14B | 60 | 4 |

---

## 五、总结与趋势

1. **注意力机制**：GQA 已成为标配，Flash Attention 类算法将持续优化对超长上下文的支持
2. **激活函数**：SwiGLU 已形成事实标准，短期内难有替代
3. **MoE**：Token 路由策略从"均衡路由"向"自适应/学习型路由"演进，细粒度专家成为趋势
4. **长上下文**：RoPE 扩展方法日趋成熟，百万级 token 上下文正在成为现实
5. **推理优化**：架构层面的 MLA（DeepSeek）与系统层面的推测解码（Speculative Decoding）正在收敛

---

## 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [DeepSeek-V2: Mixture of Experts](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Flash Attention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

---

*本文由 Coze 智能体「知识库管家」生成并持续更新 · 最后更新: 2026-05-15*
