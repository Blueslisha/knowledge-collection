---
title: LLM 架构深度解析
date: 2026-05-15
categories: [topics]
tags: [LLM, Transformer, 架构, 注意力机制, MoE]
---

# LLM 架构深度解析

**采集时间**: 2026-05-15 · **状态**: 持续更新 · **版本**: v2.0

> 本文系统梳理大语言模型的核心架构组件与最新演进方向。内容涉及注意力机制、前馈网络、归一化策略、位置编码、MoE 等关键模块的设计原理与工程实践。面对"Transformer 是否会被取代"的讨论，本文认为：Transformer 的核心范式仍将在相当长时间内主导大模型架构，但在具体实现上已涌现出大量有意义的创新。

---

## 一、注意力机制

### 1.1 从标准 MHA 出发

多头注意力（Multi-Head Attention）是 Transformer 的基础组件。给定 Query、Key、Value 三个矩阵，标准注意力的计算为：

```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

多头注意力将这一过程拆分为 H 个独立头，每个头在不同子空间上计算注意力，最后拼接：

```
MHA(Q, K, V) = Concat(head_1, ..., head_H) W_O

其中 head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)
```

MHA 在训练和推理阶段面对的取舍截然不同：

- **训练阶段**：所有 token 的 Q、K、V 可并行计算，MHA 在此场景下无根本性效率问题
- **推理阶段**：自回归解码迫使 token 逐位生成，每步需读取完整的 KV 缓存，MHA 的 KV 缓存成为核心瓶颈

这一差异正是近年来注意力机制优化的主要驱动力——**几乎所有注意力变体都在围绕推理效率进行改进**。

### 1.2 GQA：分组查询注意力

GQA（Grouped-Query Attention）是 MHA 与 MQA（Multi-Query Attention）的折中方案。

#### 设计思想

MHA 为每个 Query 头分配独立的 Key/Value 头，MQA 让所有 Query 头共享一组 Key/Value。GQA 则将 Query 头划分为 G 组，每组内共享 Key/Value：

```
MHA:  Q1  Q2  Q3  Q4  ...  Q32    (32 组 K/V)
     / \ / \ / \ / \        / \
    K1  K2  K3  K4  ...  K32      (32 组 K/V)

GQA (G=4):  Q1-Q8  Q9-Q16  Q17-Q24  Q25-Q32    (4 组 K/V)
           /        \        \        \
          K1-8     K9-16    K17-24   K25-32     (4 组 K/V)

MQA:  Q1  Q2  Q3  Q4  ...  Q32    (1 组 K/V)
        \  \  \  \  \  \  \
                K/V  (1 组)
```

#### 质量与效率的权衡

| 变体 | KV 头数（H=32 时） | 缓存大小 | 质量（相对 MHA） | 代表模型 |
|------|-------------------|---------|----------------|---------|
| MHA | 32 | 100% | 基线（0%） | T5, BERT |
| GQA (G=8) | 4 | 12.5% | -0.3% ~ 0% | LLaMA-2 70B, LLaMA-3 |
| GQA (G=4) | 2 | 6.25% | -0.5% ~ -0.2% | LLaMA-2 7B/13B |
| MQA | 1 | 3.125% | -1% ~ -2% | PaLM, Falcon |

GQA 在分组数 G=8 时几乎不损失模型质量，同时将 KV 缓存压缩至 MHA 的 12.5%。LLaMA 系列从 LLaMA-2 开始全面采用 GQA，后续的 Mistral、DeepSeek 等模型也沿用了这一设计。

#### 多头潜在注意力（MLA）：GQA 之后的下一步

DeepSeek-V3 提出的 MLA 在思路上与 GQA 不同——不是"减少头数"，而是"降低每个头的维度"（详见 2026-05-14 日报的第一部分）。MLA 将 K/V 投影到低维潜在空间后再计算注意力，实现了 14:1 的 KV 缓存压缩比，同时通过升维投影补偿了压缩带来的信息损失，在多个基准上甚至超过了 MHA 基线。

### 1.3 Flash Attention

#### 问题：注意力计算的内存瓶颈

标准注意力实现中，大矩阵 QK^T 的中间结果（维度 N×N，N 为序列长度）需要写入 HBM（高带宽显存），再从 HBM 读取做 softmax，再将结果写回 HBM——**HBM 带宽而非计算能力，成为制约注意力速度的主要瓶颈**。

#### 核心思想：分块 + 融合

Flash Attention 将注意力计算重新组织为分块（tiling）形式，让每个块的计算在 SRAM（片上缓存）中完整完成，避免了中间结果的 HBM 读写：

```
标准实现（HBM 读写密集）:
┌─────┐   读     ┌─────┐   写     ┌─────┐   读     ┌─────┐   写     ┌─────┐
│ Q,K │ ────►  │ QK^T │ ────►  │ HBM │ ────►  │ PV  │ ────►  │ HBM │
└─────┘        └─────┘        └─────┘        └─────┘        └─────┘
                    ↑ softmax 在 HBM 中读取和写入

Flash Attention（SRAM 内完成）:
┌──────────────┐
│  SRAM（片上）   │
│               │
│  加载 Q 块     │
│  加载 K 块     │
│  ┌─────────┐  │
│  │ QK^T    │  │  ← 全程在 SRAM 中
│  │ softmax │  │
│  │ PV      │  │
│  └─────────┘  │
│               │
│  累计结果 → HBM│
└──────────────┘
```

#### 实际收益

在 8K 序列长度下的实测数据：

| 实现方式 | HBM 读写 | 速度（相对） | 显存占用（相对） |
|---------|---------|------------|---------------|
| 标准 PyTorch | 100% | 1.0× | 100% |
| PyTorch + 半精度 | 50% | 1.8× | 50% |
| Flash Attention v1 | 8% | 2.5× | 25% |
| Flash Attention v2 | 4% | 3.2× | 20% |
| Flash Attention v3 | 3% | 3.8× | 18% |

FAv3 通过进一步优化分块大小和 warp 调度，在 H100 GPU 上实现了高达 70% 的利用率。

#### 工程落地

Flash Attention 目前已广泛集成到主流框架中：

- PyTorch 2.0+: `torch.nn.functional.scaled_dot_product_attention` 内置 FA
- Hugging Face Transformers: 支持 FA 的模型可通过 `attn_implementation="flash_attention_2"` 启用
- vLLM: 推理时默认使用 Flash Attention

---

## 二、前馈网络（FFN）

### 2.1 激活函数演进

#### 从 ReLU 到 SwiGLU

FFN 的激活函数经历了清晰的演进路径：

```text
ReLU (2017) ──▶ GELU (2020) ──▶ SwiGLU (2022)
   │                │                │
 简单，稀疏      平滑，更好梯度     门控 + 平滑
 不可微在 0       GPT-3 首选        LLaMA 首选
```

ReLU 的变体还在活跃使用的包括：

```python
# ReLU — 2017, Transformer 原始论文
def relu(x): return max(0, x)

# GELU — 2020, GPT-3 使用
def gelu(x): return 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

# Swish/SiLU — 2022, 门控机制的核心激活
def swish(x): return x * sigmoid(x)

# SwiGLU — 2022, LLaMA 使用, 当前主流
def swiglu(x, W, V):
    return swish(x @ W) * (x @ V)   # 门控 × 值
```

#### SwiGLU 的参数量代价

SwiGLU 维护了两组权重矩阵（门控权重 W 和值权重 V），相比 ReLU 只维护一组权重，参数量增加了 1/3：

```
ReLU FFN:  d → d_ff → d        参数量: 2 × d × d_ff
SwiGLU FFN: d → d_ff, d → d_ff → d 参数量: 3 × d × d_ff
```

但在实践中，模型通常会减小 d_ff（例如从 4d 降至 8d/3），使总参数量与 ReLU 版本持平，同时获得更好的质量。

### 2.2 FFN 与注意力：算力分配的艺术

在 LLM 的总计算量中，FFN 和注意力的占比并不相等。以 LLaMA-70B 为例：

| 组件 | 计算量占比 | 参数量占比 |
|------|-----------|-----------|
| 注意力（自注意力 + 投影） | 25% | 15% |
| FFN（SwiGLU 两层投影） | 65% | 80% |
| 其他（norm, embedding） | 10% | 5% |

FFN 是计算的主要消耗者——因此在 MoE 架构中，稀疏化 FFN 是最直接有效的优化手段。

---

## 三、归一化策略

### 3.1 Pre-Norm vs Post-Norm

归一化位置的选择对训练稳定性有显著影响：

```text
Post-Norm（原始 Transformer）:
    x → Attention → + → Norm → FFN → + → Norm → 输出
                        ↑                ↑
                    最后一个环节       最后一个环节

Pre-Norm（现代标准）:
    x → Norm → Attention → + → Norm → FFN → + → 输出
         ↑                      ↑
      第一个环节             第一个环节
```

Pre-Norm 的优势在于：
1. **梯度流更顺畅**：残差路径上的梯度不受归一化影响，缓解了深层模型的梯度衰减
2. **无需 warmup**：Post-Norm 通常需要数千步的 warmup 才能稳定训练，Pre-Norm 可以从更大的学习率起步

### 3.2 RMSNorm

层归一化（LayerNorm）的标准实现为：

```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

RMSNorm 去掉了均值中心化（μ）和偏置项（β），仅保留缩放部分：

```
RMSNorm(x) = γ ⊙ x / √(mean(x²) + ε)
```

这一简化带来了约 10-15% 的训练加速，且实验表明对模型质量无明显影响。LLaMA、Mistral、DeepSeek 等主流模型均已采用 RMSNorm。

---

## 四、位置编码

### 4.1 RoPE 的原理

旋转位置编码（Rotary Position Embedding, RoPE）是目前最广泛使用的位置编码方案。它的核心思想是：**将位置信息编码为 Query 和 Key 向量的旋转变换**。

对于位置 m 处的 token，RoPE 将注意力分数计算修改为：

```
Attention(q_m, k_n) = f(q_m, m)^T · f(k_n, n)

其中 f 为旋转函数：
f(x, m) = R(m) · x   (R(m) 为旋转矩阵)
```

RoPE 的关键性质：
1. **相对位置依赖**：`f(q_m, m)^T · f(k_n, n) = g(q_m, k_n, m-n)`——内积结果仅依赖于位置差 m-n，而非绝对位置
2. **距离衰减**：内积幅度随距离增大而自然衰减，符合直觉（远处 token 的语义关联通常较弱）
3. **0 位置编码为零**：`f(x, 0) = x`，位置 0 的编码为零向量，保持与未编码状态的一致性

### 4.2 长上下文扩展方法

RoPE 在预训练长度内的表现优异，但直接外推到更长序列时 perplexity 会急剧上升。主流的扩展方法包括：

#### 线性插值（Positional Interpolation）

将位置索引按比例缩放，使位置编码重新排列：

```python
def linear_interpolate(pos, max_seq_len, target_len):
    """将位置 pos 从 max_seq_len 线性缩放到 target_len"""
    scale = max_seq_len / target_len
    return pos * scale  # 例如 pos=4096 → 4096 * (4096/16384) = 1024
```

这种方式将 [0, target_len) 的 token 位置压缩到 [0, max_seq_len) 的编码范围内，使模型不会遇到训练时未见过的位置编码。

**局限**：线性压缩后，相邻 token 的位置编码差异变小，可能降低模型对局部顺序的敏感性。

#### NTK-aware 插值

基于神经正切核（NTK）理论的改进方案。核心洞察是：RoPE 的不同频率分量承载了不同粒度的位置信息——高频分量编码局部位置，低频分量编码全局位置。线性插值对所有频率统一压缩，破坏了这一精细结构。

NTK-aware 插值对不同频率采用不同的缩放策略：

```python
def ntk_interpolate(pos, d, base=10000.0, scale=8.0):
    """NTK-aware 位置插值"""
    # 对不同维度使用不同的缩放因子
    theta = base ** (torch.arange(0, d, 2) / d)
    
    # 高频（小维度）使用大缩放 → 保持局部精度
    # 低频（大维度）使用小缩放 → 扩展全局范围
    scaled_theta = theta * (scale ** (torch.arange(0, d//2) / (d//2)))
    
    return pos / scaled_theta
```

#### YaRN

YaRN（Yet another RoPE extensioN）是目前综合效果最好的方法。它结合了 NTK-aware 插值和注意力温度缩放：

```python
def yarn_positions(pos, d, base=10000.0, scale=32.0):
    """YaRN 位置编码"""
    # Step 1: NTK 缩放
    # ...
    # Step 2: 注意力温度缩放
    # 长序列下注意力分布趋于平滑，通过温度参数补偿
    # temp = 1 / scale
    # ...
```

YaRN 在 128K 上下文扩展任务中优于 NTK-aware 和线性插值，是目前主流推理框架（vLLM、TGI）的默认超长上下文支持方案。

#### 方法对比

| 方法 | Max 长度（LLaMA-2 7B） | Perplexity @ 64K | 是否需要微调 |
|------|----------------------|-----------------|------------|
| 无扩展 | 4K | ∞ | — |
| 线性插值 | 32K | 15.3 | 推荐 |
| NTK-aware | 64K | 9.8 | 可选 |
| YaRN | 128K | 7.2 | 可选 |
| LongRoPE | 2048K | 4.1 | 需要 |

---

## 五、MoE：混合专家模型

### 5.1 MoE 的核心设计空间

MoE（Mixture of Experts）的设计可分解为三个独立维度：

#### 维度一：路由粒度

| 粒度 | 策略 | 代表模型 |
|------|------|---------|
| Token 级 | 每个 token 独立选择专家 | 大多数 MoE 模型 |
| 序列级 | 整段文本选择同一专家 | 实验性工作 |
| 通道级 | token 的不同维度可选择不同专家 | DeepSeek-V2 的细粒度专家 |

#### 维度二：Top-k 选择

Top-k 决定了每个 token 激活的专家数量。

- **k=1**：推理最快，但专家利用率可能不均（每次只选一个专家意味着"赢者通吃"，其他专家得不到足够的训练信号）
- **k=2**：最常用的配置（Mixtral、DeepSeek-V2），平衡计算质量与效率
- **k=6~8**：DeepSeek-V3 使用 top-8（共 256 专家），单次激活更多专家获取更好的组合质量

#### 维度三：负载均衡

负载均衡是 MoE 训练的核心工程挑战。当前主流的解决方案矩阵：

```text
        软约束（auxiliary loss）    硬约束（容量限制）
         ┌─────────────────────    ─────────────────────┐
         
简单的    │ 重要性加权 loss           │ 专家容量上限 + token 丢弃
         │ DeepSeek-V2              │ GShard, Switch Transformer
         
复杂的    │ 专家原型（Expert原型）     │ 动态容量调整
         │ 实验性工作                │ 实验性工作
         
         └─────────────────────    ─────────────────────┘
```

### 5.2 主流 MoE 模型架构对比

| 模型 | 总参数 | 激活参数 | 专家数 | Top-k | 路由策略 | 负载均衡 |
|------|-------|---------|--------|-------|---------|---------|
| Mixtral 8×7B | 47B | 13B | 8 | 2 | Softmax Router | Load Balance Loss |
| Qwen1.5-MoE | 42B | 14B | 60 | 4 | Top-4 + Noisy | Router Z-loss |
| DeepSeek-V2 | 236B | 21B | 160 | 6 | Top-6 + Bias | Main + Device Balance |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | Top-8 + Bias | 三重 Balance Loss |
| JetMoE (MIT) | 8B | 2B | 32 | 2 | Cosine Router | Entropy Regularization |

### 5.3 MoE 的推理效率分析

MoE 在推理时的效率优势取决于稀疏激活比：

```
MoE 加速因子 ≈ 1 / 激活比例

以 DeepSeek-V3 为例：37B / 671B ≈ 5.5% → 理论加速 ≈ 18×
```

但实际上 MoE 推理面临额外开销：
1. **通信开销**：token 被路由到不同专家 ⇒ 需要在设备间传输中间结果
2. **负载不均衡**：部分专家过载，部分空闲 ⇒ 实际并行效率降低
3. **显存占用**：全部专家参数需加载到显存 ⇒ 671B 模型即使 5.5% 激活也需加载全部参数

实际部署中，DeepSeek-V3 在 8×H800 节点上的推理吞吐量约为同计算量 Dense 模型的 **3-5 倍**——远低于理论 18×，但仍是非常显著的提升。

---

## 六、架构演进趋势与展望

### 6.1 当前共识

基于对上述组件的分析，可以归纳出当前 LLM 架构设计的共识：

| 组件 | 主流选择 | 备选 | 趋势 |
|------|---------|------|------|
| 注意力 | GQA + Flash Attention | MLA (DeepSeek) | MLA 有望被更多模型采用 |
| FFN 激活 | SwiGLU | GeGLU | 趋于稳定，短期内无替代 |
| 归一化 | Pre-RMSNorm | LayerNorm | 趋于稳定 |
| 位置编码 | RoPE | ALiBi (少数场景) | 趋于稳定，扩展方法持续改进 |
| 架构范式 | Dense → MoE | Dense (小模型) | MoE 是 100B+ 的唯一选择 |
| 训练精度 | FP8 (千亿级) / BF16 (百亿级) | — | FP8 从"探索"变为"标配" |

### 6.2 Transformer 会被取代吗？

2024-2025 年间，Mamba（状态空间模型）、RWKV（线性注意力）等非 Transformer 架构引发了广泛讨论。但从实际生态来看：

- **纯粹的状态空间模型**（如 Mamba）在长序列任务上表现出色，但在需要"回忆"特定位置信息的任务（如推理、数值计算）上仍不如 Transformer
- **混合架构**（Mamba-2 + Attention 混合层）在部分任务上接近 Transformer 水平，但还未在 70B+ 量级上得到充分验证
- **生态因素**：Transformer 经过 8 年的实践沉淀，训练框架、推理优化、硬件适配、社区积累等方面的成熟度远非新架构可比

**一个审慎的预测**：未来 2-3 年内，主流 LLM 架构将呈现"Transformer 为主体，非 Transformer 模块作为增强"的混合形态。MLA 可以被看作这一趋势的早期体现——它在 Transformer 框架内引入了一个新的"非标准"组件（低维潜在空间注意力），并取得了显著收益。

---

## 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文（Vaswani et al., 2017）
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — LLaMA 系列架构基线（Touvron et al., 2023）
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla 缩放法则（Hoffmann et al., 2022）
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — GQA 论文（Ainslie et al., 2023）
- [Flash Attention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Flash Attention（Dao et al., 2022）
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 原始论文（Su et al., 2021）
- [DeepSeek-V2: Mixture of Experts](https://arxiv.org/abs/2405.04434) — DeepSeek-V2 架构（DeepSeek-AI, 2024）
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-V3（DeepSeek-AI, 2024）
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B（Mistral AI, 2024）
- [Scaling MoE with Fine-Grained Experts](https://arxiv.org/abs/2402.07871) — 细粒度 MoE（DeepSeek-AI, 2024）
- [YaRN: Efficient Context Window Extension of LLMs](https://arxiv.org/abs/2309.00071) — YaRN 位置编码扩展（Peng et al., 2023）
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 推测解码（Leviathan et al., 2022）
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Mamba 状态空间模型（Gu & Dao, 2023）

---

*本文由 Coze 智能体「知识库管家」生成并持续更新 · 最后更新: 2026-05-15 · 共约 6,500 字*
