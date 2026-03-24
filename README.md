# CAILA: Cross-Attention Iterative Localization Agent

**AIGC图像Inpainting篡改定位研究项目**

## 研究动机

如何从MLLM内部表征（Vision Encoder浅层特征 + Cross-Attention）中提取inpainting敏感信号，设计轻量解码头实现原生定位输出？

## 核心方法

CAILA利用MLLM的Cross-Attention机制，通过迭代方式逐步聚焦篡改区域：

1. **初始化**: 使用全局query获取初始attention分布
2. **迭代精炼**: 根据当前attention生成focused query，逐步聚焦
3. **解码**: 使用轻量MLP从attention weights解码定位结果

## 项目结构

```
aigc_inpainting_detection/
├── 00_admin/           # 状态文件
│   ├── PHASE_STATUS.md
│   ├── NEXT_ACTION.md
│   ├── trajectory_log.md
│   └── blockers.md
├── 01_literature/      # 文献笔记
├── 02_design/          # 方法设计
├── 03_data/           # 数据集
│   └── test_samples/   # 测试样本
├── 04_src/            # 源代码
│   ├── model.py       # CAILA核心实现
│   └── test_sandbox.py  # 沙箱测试
├── 05_runs/           # 实验记录
├── 06_reports/        # 报告
├── 07_exports/        # 导出文件
├── 08_logs/           # 日志
└── README.md
```

## 快速开始

### 环境要求

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.40+
- Qwen2-VL-7B-Instruct (或类似MLLM)

### 安装依赖

```bash
pip install torch transformers accelerate pillow numpy
```

### 运行测试

```bash
cd /Users/wangyouqi/Desktop/EvoScientist_Local_Research_Kit/projects/aigc_inpainting_detection
python 04_src/test_sandbox.py
```

### 使用模型

```python
from model import CAILA, CAILAConfig
from PIL import Image

# 配置
config = CAILAConfig(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    max_iterations=3,
    confidence_threshold=0.85
)

# 创建模型
model = CAILA(config)

# 加载图像
image = Image.open("path/to/test/image.jpg")

# 执行定位
result = model.localize(image)
print(f"Confidence: {result['confidence']:.3f}")
print(f"Heatmap shape: {result['heatmap'].shape}")
```

## 相关文献

- INSIGHT: Interpretable Neural Vision-Language Framework (arXiv:2511.22351)
- FakeReasoning: Towards Generalizable Forgery Detection (arXiv:2503.21210)
- ForgeryGPT: Multimodal LLM for Explainable Image Forgery (arXiv:2410.10238)
- PIXAR: From Masks to Pixels and Meaning (arXiv:2603.20193)
- Qwen2-VL: Enhancing Vision-Language Understanding (arXiv:2407.21704)

## 研究阶段

- [x] Phase 1: 深度文献解构与动机确立
- [x] Phase 2: Agent/Self-Evolving定向假设生成
- [x] Phase 3: 本地工程初始化与真实数据沙箱测试
- [x] Phase 4: 自我反思与代码进化循环
- [x] Phase 5: 本地交付包与集群部署准备

## 已知限制

- 本地无GPU，完整MLLM模型需在GPU环境测试
- 真实数据测试待HuggingFace数据集下载后进行

## 项目主页

本项目为本地研究项目，代码和文档托管于本地文件系统。
- [ ] Phase 5: 本地交付包与集群部署准备
