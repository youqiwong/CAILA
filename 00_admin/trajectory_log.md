# Trajectory Log

## 研究任务概述
- **任务方向**: AIGC图像局部篡改定位与检测
- **聚焦问题**: Inpainting篡改定位与检测
- **方法约束**: 利用MLLM原生能力，不依赖外部分割模型
- **目标模型**: Qwen3-VL-4B-Instruct等MLLM

## 时间线

### 2026-03-24 - 阶段1完成
- **Phase**: Phase 1 - 深度文献解构与动机确立
- **Status**: completed
- **Action**: 
  - 调研了2025-2026年MLLM图像取证最新研究
  - 分析了BLINK、PIXAR、ForgeryGPT、FakeReasoning等关键工作
  - 确立了核心研究动机：从MLLM内部表征提取inpainting敏感信号

### 关键文献发现
1. BLINK基准显示GPT-4V在细粒视觉任务上仅51%准确率 vs 人类95%
2. 现有方法依赖外部DINO/SAM，缺乏对MLLM内部表征的利用
3. Vision encoder浅层特征包含丰富的artifacts信息但在投影后丢失
4. Cross-attention机制可提供查询-视觉区域对应关系

### 核心动机确立
**问题**: 如何从MLLM内部表征（Vision Encoder浅层特征 + Cross-Attention）中提取inpainting敏感信号，设计轻量解码头实现原生定位输出？

---

### 2026-03-24 - 阶段2完成
- **Phase**: Phase 2 - 基于Agent/Self-Evolving的定向假设生成
- **Status**: completed
- **Action**:
  - 设计了3个方向: AMLF, CAILA, VSDCL
  - 选择CAILA作为主线方案
  - CAILA: 交叉注意力迭代定位Agent，直接利用MLLM的Cross-Attention机制进行inpainting定位

### 方案选择理由
1. **CAILA**: 直接利用Cross-Attention，动机契合度最高，理论创新突出
2. **AMLF放弃**: 工程简单但创新有限
3. **VSDCL放弃**: 复杂度过高，偏离轻量设计原则

---

### 2026-03-24 - 项目初始化
- **Phase**: Phase 1 - 深度文献解构与动机确立
- **Status**: in_progress
- **Action**: 开始文献调研
