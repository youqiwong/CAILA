# 阶段1文献调研：MLLM图像篡改定位研究诊断

## 研究概述
- **任务方向**: AIGC图像局部篡改定位与检测（聚焦Inpainting）
- **约束**: 利用MLLM原生能力，不依赖外部分割模型
- **目标模型**: Qwen3-VL-4B-Instruct等MLLM

---

## 1. 2025-2026年MLLM图像取证研究现状

### 1.1 现有方法分类

| 类别 | 代表工作 | 方法特点 | 局限性 |
|------|----------|----------|--------|
| **可解释检测** | INSIGHT, ForenX, ForgeryGPT | 端到端MLLM + 链式推理 | 依赖外部视觉编码器特征 |
| **双编码器融合** | FakeReasoning | CLIP(高层语义) + DINO(低层 artifacts) | 仍需外部DINO特征 |
| **知识引导** | Unlocking LVLMs | 知识图谱 + 提示学习 | 泛化性受限 |
| **编辑定位** | EditScout, DiffSeg30k | 扩散编辑定位 | 未聚焦Inpainting细粒度 |

### 1.2 BLINK基准揭示的核心问题

**关键发现** (BLINK Benchmark, arXiv:2404.12390):
- 人类基线准确率: **95.70%**
- GPT-4V准确率: **51.26%** (仅比随机高13.17%)
- Gemini准确率: **45.72%** (仅比随机高7.63%)

**结论**: MLLM的核心视觉感知能力尚未"涌现"，在细粒度任务上存在显著差距。

### 1.3 当前方案核心缺陷

1. **外部依赖过重**
   - FakeReasoning依赖DINO特征
   - ForgeryGPT依赖Mask Encoder
   - 大多数方法依赖SAM进行分割输出

2. **表征利用不足**
   - 现有方法主要使用MLLM的语言推理能力
   - 忽视MLLM内部视觉表征（attention、hidden states、visual tokens）
   - 未充分挖掘vision-language对齐前的原始视觉特征

3. **定位精度问题**
   - PIXAR (arXiv:2603.20193) 揭示: 基于mask的指标导致严重的过评分/欠评分
   - MLLM的坐标幻觉(coordinate hallucination)问题
   - 缺乏像素级的精确输出机制

4. **跨生成器泛化**
   - 大多数检测器针对特定生成器训练
   - 在新生成器上性能显著下降
   - DiffSeg30k表明分割模型比分类器泛化更好

---

## 2. 核心缺陷深度分析

### 2.1 细粒度感知缺陷

MLLM在以下场景表现不足:
- 小尺寸篡改区域 (< 32x32 pixels)
- 复杂遮挡场景
- 细微纹理差异
- 语义一致性与低层artifacts的联合判断

### 2.2 表征提取现状

| 表征类型 | 提取难度 | 用于篡改检测的潜力 | 现有利用程度 |
|----------|----------|-------------------|--------------|
| Visual Tokens (投影后) | 低 | 中 | 高 (主要用于VL对齐) |
| Cross-Attention Maps | 中 | 高 | 低 |
| Hidden States | 中 | 高 | 极低 |
| Vision Encoder Features (浅层) | 中 | 极高 | 极低 |
| Query/Key/Value activations | 高 | 极高 | 无 |

### 2.3 最值得突破的缺陷

**关键洞察**: 
- 当前研究聚焦于"如何用MLLM的推理能力"，而非"如何从MLLM内部提取细粒度视觉信号"
- Vision encoder的浅层特征包含丰富的纹理和artifacts信息，但这些信息在投影到LLM空间后大量丢失
- Cross-attention机制理论上可以揭示语言查询与视觉区域的对应关系

---

## 3. 唯一、尖锐、可验证的研究动机

### 3.1 研究动机陈述

**核心问题**: 如何从MLLM内部表征（特别是Vision Encoder浅层特征和Cross-Attention）中，提取对Inpainting篡改敏感的细粒度空间信号，设计轻量解码头实现原生定位输出？

### 3.2 动机来源

1. **理论必要性**: MLLM的vision encoder浅层保留了丰富的低层视觉特征（纹理、边缘、噪声分布），这些是检测inpainting artifacts的关键信号，但当前方法在投影到LLM空间时丢失了这些信息。

2. **工程可行性**: Qwen-VL等模型的attention机制提供了查询-视觉区域的对应关系，可用于定位。

3. **方法论缺口**: 现有工作要么使用外部分割模型（SAM），要么完全依赖LLM推理，均未充分利用MLLM内部的多粒度表征层次。

### 3.3 可验证性

- **基线对比**: 直接从Qwen-VL attention提取的定位结果 vs 本文方法
- **消融实验**: 验证各表征层（浅层/中层/深层/attention）的贡献
- **标准数据集**: CASIA, Columbia, NIST等inpainting检测数据集

---

## 4. 文献表

### 核心引用

| # | 论文 | arXiv ID | 关键贡献 |
|---|------|----------|----------|
| 1 | INSIGHT | 2511.22351 | 可解释AI生成图像检测 |
| 2 | ForenX | 2508.01402 | 多模态LLM forgery检测 |
| 3 | ForgeryGPT | 2410.10238 | Mask-aware forgery extractor |
| 4 | FakeReasoning | 2503.21210 | CLIP+DINO双分支 |
| 5 | EditScout | 2412.03809 | 扩散编辑定位 |
| 6 | DiffSeg30k | 2511.19111 | 30K扩散编辑定位基准 |
| 7 | PIXAR | 2603.20193 | 像素级篡改taxonomy |
| 8 | BLINK | 2404.12390 | MLLM视觉感知基准 |
| 9 | RealHD | 2602.10546 | 高质量AI生成图像检测数据集 |
| 10 | Unlocking LVLMs | 2503.14853 | 知识引导deepfake检测 |
| 11 | Qwen3-VL | 2511.21631 | Qwen3-VL架构 |
| 12 | ChatGPT Splicing | 2506.05358 | GPT-4V零样本拼接检测 |
| 13 | LLMs Not Ready | 2506.10474 | LLM deepfake检测局限性分析 |
| 14 | LatentGeo | 2603.12166 | 潜在表征空间几何构建 |
| 15 | NavMind | 2603.21577 | MLLM心理导航能力 |

---

## 5. 研究空白与机会

### 5.1 已验证的空白

1. **MLLM内部表征挖掘不足**: attention maps、hidden states未被用于inpainting检测
2. **轻量解码头缺失**: 缺乏直接从MLLM token space到空间定位的轻量映射
3. **跨模态注意力利用**: Cross-attention机制在篡改定位中的应用未被充分探索

### 5.2 潜在研究方向

1. **多层特征融合定位**: 融合vision encoder不同层的特征，保留高层语义的同时保留低层artifacts
2. **Attention-guided Localization**: 利用query-specific cross-attention maps进行定位
3. **Token-level轻量分类器**: 直接在visual tokens上训练轻量MLP进行二分类
4. **Self-Evolving Attention**: 迭代优化attention weights以聚焦篡改区域

---

## 6. 结论

当前MLLM图像篡改定位研究存在"外部依赖过重"和"内部表征利用不足"的双重问题。本项目提出从MLLM内部多层次表征中提取inpainting敏感信号，设计轻量解码头实现端到端定位的方法论。这一方向既有理论基础（vision encoder浅层特征的artifacts敏感性），又有工程可行性（Qwen-VL的attention机制），填补了现有研究的方法论缺口。
