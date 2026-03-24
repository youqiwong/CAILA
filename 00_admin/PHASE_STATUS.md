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
- **Status**: completed

### 阶段3完成情况
- [x] Git仓库初始化完成
- [x] 核心代码结构创建 (model.py, test_sandbox.py)
- [x] CAILA模型原型实现
- [x] 沙箱测试通过 (5/6, GPU相关跳过)
- [x] 合成测试数据生成
- [x] 首次commit完成

### 硬件限制记录
- 本地无GPU，无法加载Qwen2-VL-7B完整模型
- 沙箱测试在CPU上验证了decoder、attention hook等组件

---

## 下一阶段

- **Phase**: Phase 4 - 自我反思与代码进化循环
- **Status**: completed

### 阶段4完成情况

#### 迭代1: 空间注意力池化改进
- 添加SpatialAttentionPooling模块
- 改进decoder从mean pooling到spatial attention pooling
- 测试全部通过

#### 迭代2: 训练脚本和配置
- 添加train.py: 支持DDP、混合精度、WandB日志
- 添加config.yaml: 完整配置文件
- 添加download_data.py: 数据下载脚本
- 添加CAILALoss自定义损失函数

### 核心代码文件
- `04_src/model.py`: CAILA核心模型
- `04_src/train.py`: 训练脚本
- `04_src/config.yaml`: 配置文件
- `04_src/test_sandbox.py`: 测试脚本

---

## 下一阶段

- **Phase**: Phase 5 - 本地交付包与集群部署准备
- **Status**: in_progress
