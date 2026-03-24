# CAILA技术报告

## 研究摘要

**项目**: CAILA - Cross-Attention Iterative Localization Agent  
**任务**: AIGC图像Inpainting篡改定位  
**核心动机**: 从MLLM内部表征中提取inpainting敏感信号，设计轻量解码头实现原生定位输出

---

## 1. 研究背景与动机

### 1.1 问题定义

AIGC（AI-Generated Content）图像篡改检测是当前数字取证领域的重要挑战。其中，Inpainting（图像修复/填充）篡改因其生成自然、无明显痕迹而特别难以检测。

### 1.2 现有方案缺陷

基于文献调研（2025-2026年最新研究），现有方案存在以下核心缺陷：

1. **外部依赖过重**: 现有方法（如FakeReasoning、ForgeryGPT）依赖SAM、DINO等外部分割模型，缺乏对MLLM内部表征的利用

2. **内部表征利用不足**: BLINK基准测试显示GPT-4V在细粒视觉任务上仅51%准确率（vs 人类95%），说明MLLM的内部表征包含未被挖掘的空间信息

3. **定位精度问题**: MLLM坐标幻觉，缺乏像素级精确输出

### 1.3 研究问题

**核心问题**: 如何从MLLM内部表征（Vision Encoder浅层特征 + Cross-Attention）中提取对Inpainting篡改敏感的细粒度空间信号，设计轻量解码头实现原生定位输出？

---

## 2. 方法设计

### 2.1 方案选择

从三个候选方向中选择**CAILA (Cross-Attention Iterative Localization Agent)**:

| 方案 | 核心思想 | Agent/Self-Evolving要素 | 选择理由 |
|------|---------|----------------------|---------|
| AMLF | 多层特征融合 | 自适应层选择 | 创新有限 |
| **CAILA** | Cross-Attention迭代定位 | 动态query生成 | **动机契合度最高** |
| VSDCL | 双路对比学习 | 对比机制 | 复杂度高 |

### 2.2 CAILA核心架构

```
输入图像
    ↓
┌─────────────────────────────┐
│   Vision Encoder (Qwen-VL)   │
│   - 提取视觉token表征        │
│   - 保留浅层纹理特征          │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│   Cross-Attention Hook       │
│   - 提取query-visual attention│
│   - 定位异常区域              │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│   Query Generator            │
│   - 根据当前状态生成focused query│
│   - 迭代聚焦篡改区域          │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│   Spatial Attention Decoder  │
│   - 空间注意力池化            │
│   - 轻量MLP解码               │
│   - 输出14x14定位heatmap      │
└─────────────────────────────┘
    ↓
定位heatmap + 置信度
```

### 2.3 核心组件

#### 2.3.1 SpatialAttentionPooling

```python
class SpatialAttentionPooling(nn.Module):
    """空间注意力池化 - 保留更多空间信息"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
```

#### 2.3.2 CAILADecoder

```python
class CAILADecoder(nn.Module):
    """轻量化解码头"""
    
    def forward(self, visual_features):
        # 空间注意力池化
        attn_weights = self.attention(visual_features)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权求和
        pooled = (visual_features * attn_weights).sum(dim=1)
        
        # MLP解码
        logits = self.mlp(pooled)
        return torch.sigmoid(logits.view(-1, 1, 14, 14))
```

#### 2.3.3 QueryGenerator

动态生成focused query，支持迭代精炼：

```python
class QueryGenerator(nn.Module):
    """动态Query生成器"""
    
    def forward(self, state_emb, stage):
        # 根据当前状态和阶段生成query
        stage_emb = self.stage_embeddings[stage]
        query = self.state_encoder(state_emb) + stage_emb
        return query
```

---

## 3. 实验设置

### 3.1 环境配置

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **Transformers**: 4.40+
- **基础模型**: Qwen2-VL-7B-Instruct
- **硬件**: 本地测试CPU；GPU训练需≥14GB VRAM

### 3.2 训练配置

```yaml
training:
  epochs: 10
  batch_size: 4
  learning_rate: 1.0e-4
  gradient_accumulation_steps: 4
  use_amp: true

loss:
  bce_weight: 1.0
  dice_weight: 0.5
  iou_weight: 0.5
```

### 3.3 沙箱测试结果

| 测试项 | 状态 | 说明 |
|--------|------|------|
| imports | PASS | 模块导入成功 |
| initialization | PASS | 模型组件初始化成功 |
| decoder_forward | PASS | Decoder输出正确 |
| attention_hook | PASS | Attention Hook正常 |
| synthetic_data | PASS | 生成4个测试样本 |
| model_load | SKIP | 需要GPU |

---

## 4. 代码结构

```
aigc_inpainting_detection/
├── 04_src/
│   ├── model.py           # CAILA核心模型
│   ├── train.py           # 训练脚本 (DDP/AMP/WandB)
│   ├── config.yaml         # 配置文件
│   └── test_sandbox.py    # 沙箱测试
├── 03_data/
│   ├── test_samples/      # 测试样本
│   └── download_data.py   # 数据下载脚本
└── README.md
```

---

## 5. 使用指南

### 5.1 环境安装

```bash
pip install torch transformers accelerate pillow numpy
pip install wandb  # 可选，用于日志
```

### 5.2 单图推理

```python
from model import CAILA, CAILAConfig
from PIL import Image

config = CAILAConfig(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    max_iterations=3
)
model = CAILA(config)
model.load_model()

image = Image.open("test.jpg")
result = model.localize(image)
print(f"Confidence: {result['confidence']:.3f}")
```

### 5.3 训练

```bash
python train.py \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --epochs 10 \
    --batch_size 4 \
    --use_wandb \
    --output_dir ./runs
```

### 5.4 分布式训练

```bash
torchrun --nproc_per_node=4 train.py \
    --world_size 4 \
    --use_wandb
```

---

## 6. 局限性与未来工作

### 6.1 当前局限

1. **硬件限制**: 完整模型需GPU环境，本地仅完成CPU沙箱测试
2. **数据限制**: 尚未在真实Inpainting数据集上验证
3. **迭代次数**: 当前设计支持3次迭代，可根据实际效果调整

### 6.2 未来工作

1. **GPU验证**: 在A100/V100等GPU上进行完整模型测试
2. **数据集**: 使用DiffSeg30k、PIXAR等基准数据集进行评测
3. **迭代优化**: 调整QueryGenerator实现更精准的迭代聚焦
4. **多尺度输出**: 支持更高分辨率的定位输出

---

## 7. 结论

本研究提出了CAILA框架，通过利用MLLM的Cross-Attention机制实现AIGC图像Inpainting篡改定位。核心贡献包括：

1. **动机确立**: 明确指出现有方法的外部依赖问题
2. **架构设计**: CAILA通过迭代query生成和空间注意力池化实现轻量定位
3. **代码实现**: 提供完整的训练和推理代码
4. **扩展性**: 支持DDP分布式训练和WandB日志

后续工作将聚焦于GPU环境验证和真实数据集评测。

---

## 参考文献

1. INSIGHT (arXiv:2511.22351) - Interpretable Vision-Language Framework
2. FakeReasoning (arXiv:2503.21210) - Generalizable Forgery Detection
3. PIXAR (arXiv:2603.20193) - Pixel-grounded Tampering Taxonomy
4. Qwen2-VL - Vision-Language Understanding

---

*报告生成时间: 2026-03-24*
