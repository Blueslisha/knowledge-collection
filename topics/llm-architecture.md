---
title: LLM 架构深度解析
date: 2026-05-15
categories: [topics]
tags: [LLM, Transformer, 架构]
---

# LLM 架构深度解析

**采集时间**: 2026-05-15

---

## 概述

大语言模型（LLM）的核心架构基于 Transformer 的 Decoder-only 设计。自 GPT 系列以来，这一范式主导了大模型领域的发展。本文梳理 LLM 架构的关键组件与最新演进方向。

---

## 核心组件

### 1. 多头注意力机制（Multi-Head Attention）

标准注意力计算：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

最新优化：
- **Grouped-Query Attention (GQA)**：减少 KV 头数量，降低推理时的 KV 缓存开销
- **Flash Attention**：通过分块计算和重排，显著减少显存占用

### 2. 前馈网络（FFN）

SwiGLU 已成为主流激活函数的选择，相比 ReLU 可提升约 2-3% 的下游任务表现。

### 3. 层归一化

**Pre-Norm**（在子层之前做归一化）已成为标准配置，相比 Post-Norm 训练稳定性更好。

---

## 关键架构变体

| 模型 | 参数量 | 特色 |
|------|--------|------|
| LLaMA | 7B-70B | SwiGLU + RoPE + GQA |
| DeepSeek | 7B-67B | Multi-head Latent Attention |
| Mistral | 7B | Sliding Window Attention |

---

## 最新趋势

1. **MoE（混合专家）**：通过稀疏激活大幅提升模型容量，DeepSeek-MoE 等模型验证了其有效性
2. **长上下文扩展**：RoPE 的基数放大、ALiBi 位置编码等方法持续突破上下文长度限制
3. **推理时计算**：Chain-of-Thought、Tree-of-Thought 等推理策略的架构级支持

---

## 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer 论文
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [DeepSeek-V2: Mixture of Experts](https://arxiv.org/abs/2405.04434)

---

*本文由 Coze 智能体「知识库管家」生成，内容仅供参考*
