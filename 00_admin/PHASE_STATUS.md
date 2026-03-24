# Phase Status

- **Phase**: Phase 1 - 深度文献解构与动机确立
- **Status**: completed
- **Summary**: 完成文献调研，确立核心动机：从MLLM内部多层次表征中提取inpainting敏感信号

## 文献调研结论

### 核心缺陷
1. **外部依赖过重**: 现有方法依赖SAM、DINO等外部分割模型
2. **内部表征利用不足**: attention maps、hidden states、vision encoder浅层特征未被充分挖掘
3. **定位精度问题**: MLLM坐标幻觉，缺乏像素级精确输出

### 确立的研究动机
**核心问题**: 如何从MLLM内部表征（特别是Vision Encoder浅层特征和Cross-Attention）中，提取对Inpainting篡改敏感的细粒度空间信号，设计轻量解码头实现原生定位输出？

### 拟进入阶段2的依据
1. 理论必要性: Vision encoder浅层保留了丰富的低层视觉特征，在投影到LLM空间时丢失
2. 工程可行性: Qwen-VL的attention机制提供查询-视觉区域对应关系
3. 方法论缺口: 现有工作未充分利用MLLM内部的多粒度表征层次

---

## 下一阶段

- **Phase**: Phase 2 - 基于Agent/Self-Evolving的定向假设生成
- **Status**: completed

### 选择方案
- **主线方案**: CAILA (Cross-Attention Iterative Localization Agent)
- **放弃方案**: AMLF (自适应多层特征融合), VSDCL (双路对比定位器)
- **选择理由**: CAILA直接利用Cross-Attention机制，动机契合度最高，理论创新突出

---

## 下一阶段

- **Phase**: Phase 3 - 本地工程初始化与真实数据沙箱测试
- **Status**: pending
