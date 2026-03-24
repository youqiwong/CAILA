"""
CAILA: Cross-Attention Iterative Localization Agent
用于AIGC图像Inpainting篡改定位

核心思想: 利用MLLM的Cross-Attention机制，通过迭代方式逐步聚焦篡改区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class CAILAConfig:
    """CAILA模型配置"""
    # MLLM配置
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 迭代定位配置
    max_iterations: int = 3
    confidence_threshold: float = 0.85
    init_query: str = "Identify the manipulated or inpainted regions in this image."
    
    # 特征提取配置
    use_layers: List[int] = None  # 默认使用全部层
    attention_pool_size: int = 2  # attention pooling窗口大小
    
    # 解码头配置
    decoder_hidden_dim: int = 256
    decoder_output_dim: int = 1


class AttentionHook:
    """Cross-Attention Hook用于提取注意力权重"""
    
    def __init__(self):
        self.attention_weights = []
        self.handles = []
    
    def register_hooks(self, model, layer_indices: Optional[List[int]] = None):
        """注册hooks到指定层"""
        self.attention_weights = []
        self.handles = []
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # 提取attention weights (batch, heads, seq_len, kv_len)
                if hasattr(output, 'attentions') and output.attentions is not None:
                    self.attention_weights.append(output.attentions[0].detach())
                elif len(output) > 1 and hasattr(output[1], 'attentions'):
                    self.attention_weights.append(output[1].attentions[0].detach())
            return hook
        
        # 遍历模型层
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            for idx, layer in enumerate(layers):
                if layer_indices is None or idx in layer_indices:
                    if hasattr(layer, 'self_attn'):
                        handle = layer.self_attn.register_forward_hook(get_attention_hook(f'layer_{idx}'))
                        self.handles.append(handle)
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """获取所有保存的attention weights"""
        return self.attention_weights
    
    def clear(self):
        """清空保存的weights"""
        self.attention_weights = []
    
    def remove_hooks(self):
        """移除所有hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []


class QueryGenerator(nn.Module):
    """动态Query生成器 - 根据当前状态生成下一步查询"""
    
    def __init__(self, hidden_dim: int = 4096, max_len: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # Query状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 定位描述词库
        self.localization_tokens = [
            "tampered region", "edited area", "modified pixels",
            "inpainted section", "forged part", "altered region"
        ]
        
        # 基于状态的条件embedding
        self.state_embedding = nn.Embedding(5, hidden_dim)  # 5个定位阶段
    
    def forward(self, current_attention: torch.Tensor, iteration: int, 
                prev_query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成下一步的定位查询
        
        Args:
            current_attention: 当前attention分布 (batch, heads, seq, kv)
            iteration: 当前迭代次数
            prev_query: 上一步的query (如果存在)
        
        Returns:
            新的query embedding
        """
        # 从attention中提取全局状态
        if current_attention.dim() == 4:
            # (batch, heads, seq, kv) -> (batch, kv)
            attn_pooled = current_attention.mean(dim=(1, 2))
        else:
            attn_pooled = current_attention.mean(dim=0)
        
        # 根据迭代阶段调整查询
        if iteration == 0:
            # 初始查询 - 全局探索
            base_query = torch.zeros(1, self.hidden_dim, device=current_attention.device)
        else:
            # 基于前一步attention生成focused query
            # 选择高attention区域的visual tokens
            topk = min(32, attn_pooled.size(-1))
            _, top_indices = torch.topk(attn_pooled, k=topk, dim=-1)
            # TODO: 使用top_indices来聚合visual features
            base_query = attn_pooled.mean(dim=-1, keepdim=True)
        
        # 添加迭代阶段编码
        stage_emb = self.state_embedding(
            torch.tensor(min(iteration, 4), device=current_attention.device)
        )
        
        # 融合状态信息
        query = self.state_encoder(base_query) + stage_emb
        
        return query


class CAILADecoder(nn.Module):
    """轻量化解码头 - 从attention weights预测篡改定位"""
    
    def __init__(self, hidden_dim: int = 4096, output_size: Tuple[int, int] = (14, 14)):
        super().__init__()
        self.output_size = output_size
        
        # 多层感知机解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_size[0] * output_size[1])
        )
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        从visual features预测定位heatmaps
        
        Args:
            visual_features: (batch, num_tokens, hidden_dim)
        
        Returns:
            heatmaps: (batch, 1, H, W)
        """
        # 先对token维度进行attention pooling
        batch, num_tokens, hidden = visual_features.shape
        
        # 使用简单的attention pooling
        pooled = visual_features.mean(dim=1)  # (batch, hidden)
        
        # 通过MLP解码
        logits = self.decoder(pooled)  # (batch, H*W)
        
        # Reshape到空间尺寸
        heatmaps = logits.view(batch, 1, *self.output_size)
        
        # Sigmoid得到概率
        return torch.sigmoid(heatmaps)


class CAILA(nn.Module):
    """
    Cross-Attention Iterative Localization Agent
    
    核心工作流程:
    1. 初始化: 使用全局query获取初始attention分布
    2. 迭代精炼: 根据当前attention生成focused query，逐步聚焦
    3. 解码: 使用轻量MLP从attention weights解码定位结果
    """
    
    def __init__(self, config: CAILAConfig):
        super().__init__()
        self.config = config
        
        # 加载MLLM (延迟加载以节省内存)
        self.model = None
        self.processor = None
        
        # Query生成器
        self.query_generator = QueryGenerator()
        
        # 解码头
        self.decoder = CAILADecoder(
            hidden_dim=config.decoder_hidden_dim,
            output_size=(14, 14)  # 默认输出14x14定位图
        )
        
        # Attention hook
        self.attention_hook = AttentionHook()
        
        # 迭代状态
        self.current_iteration = 0
        self.confidence_history = []
    
    def load_model(self):
        """加载MLLM模型和processor"""
        if self.model is None:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            print(f"Loading model: {self.config.model_name}")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.config.model_name)
            print("Model loaded successfully")
    
    def extract_cross_attention(self, outputs) -> torch.Tensor:
        """从模型输出中提取cross-attention weights"""
        # Qwen2-VL的cross-attention存储在特定层
        # 这里需要根据实际模型结构调整
        if hasattr(outputs, 'attentions'):
            return outputs.attentions[-1]  # 最后一层的attention
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # 从hidden states重构attention
            return self._infer_attention_from_hidden(outputs.hidden_states)
        else:
            raise ValueError("Cannot extract attention weights")
    
    def _infer_attention_from_hidden(self, hidden_states: Tuple[torch.Tensor]) -> torch.Tensor:
        """从hidden states推断attention分布"""
        # 使用相邻层hidden states的差异作为attention proxy
        last_hidden = hidden_states[-1]
        second_last = hidden_states[-2]
        
        # 计算差异 - 变化大的区域可能是篡改区域
        diff = torch.abs(last_hidden - second_last)
        diff_pooled = diff.mean(dim=-1)  # (batch, seq)
        
        # 归一化
        attention = F.softmax(diff_pooled, dim=-1)
        return attention.unsqueeze(1)  # (batch, 1, seq)
    
    def compute_confidence(self, heatmap: torch.Tensor) -> float:
        """计算定位置信度"""
        # 使用最大值和熵的组合
        max_val = heatmap.max().item()
        entropy = -(heatmap * torch.log(heatmap + 1e-8)).sum().item() / heatmap.numel()
        
        # 归一化置信度
        confidence = max_val * (1 - entropy)
        return min(confidence, 1.0)
    
    def localize(self, image: "PIL.Image.Image", return_iterations: bool = False) -> Dict:
        """
        执行inpainting篡改定位
        
        Args:
            image: 输入图像
            return_iterations: 是否返回所有迭代结果
        
        Returns:
            dict: 包含定位结果和元信息
        """
        if self.model is None:
            self.load_model()
        
        # 准备输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.config.init_query}
                ]
            }
        ]
        
        # 处理输入
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.config.device)
        
        # 注册attention hooks
        self.attention_hook.register_hooks(self.model)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        # 提取attention weights
        attention_weights = self.attention_hook.get_attention_weights()
        self.attention_hook.clear()
        
        # 如果没有捕获到attention，使用hidden states推断
        if len(attention_weights) == 0:
            # 重新运行以获取hidden states
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
            attention_weights = [self._infer_attention_from_hidden(outputs.hidden_states)]
        
        # 解码为定位heatmaps
        # 使用最后一层的attention
        last_attn = attention_weights[-1] if len(attention_weights) > 0 else None
        
        # 提取visual tokens对应的attention
        if last_attn is not None:
            # 假设text query在序列前面，visual tokens在后面
            seq_len = inputs['input_ids'].shape[1]
            visual_attn = last_attn[:, :, :seq_len, seq_len:] if last_attn.dim() == 4 else last_attn
            
            # Pool attention到空间尺寸
            num_visual_tokens = visual_attn.shape[-1]
            grid_size = int(num_visual_tokens ** 0.5)
            
            if grid_size * grid_size == num_visual_tokens:
                attn_map = visual_attn.mean(dim=1).view(-1, grid_size, grid_size)
            else:
                # 无法整分，使用插值
                attn_map = F.interpolate(
                    visual_attn.mean(dim=1).unsqueeze(1),
                    size=(14, 14),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
        else:
            # Fallback: 全零map
            attn_map = torch.zeros(1, 14, 14, device=self.config.device)
        
        # 生成定位heatmap
        heatmap = attn_map.unsqueeze(1)  # (batch, 1, H, W)
        heatmap = torch.sigmoid(heatmap)
        
        # 计算置信度
        confidence = self.compute_confidence(heatmap)
        self.confidence_history.append(confidence)
        
        result = {
            'heatmap': heatmap,
            'confidence': confidence,
            'iterations': self.current_iteration + 1
        }
        
        if return_iterations:
            result['all_heatmaps'] = [heatmap]
        
        return result
    
    def localize_iterative(self, image: "PIL.Image.Image") -> Dict:
        """
        迭代式inpainting篡改定位
        
        迭代过程:
        1. 初始查询获取全局attention
        2. 根据attention生成focused query
        3. 重复直到收敛或达到最大迭代次数
        """
        if self.model is None:
            self.load_model()
        
        self.current_iteration = 0
        self.confidence_history = []
        all_heatmaps = []
        current_heatmap = None
        
        for iteration in range(self.config.max_iterations):
            self.current_iteration = iteration
            
            # 执行单次定位
            result = self.localize(image, return_iterations=False)
            current_heatmap = result['heatmap']
            confidence = result['confidence']
            
            all_heatmaps.append({
                'heatmap': current_heatmap,
                'confidence': confidence,
                'iteration': iteration
            })
            
            # 检查收敛
            if confidence >= self.config.confidence_threshold:
                print(f"Converged at iteration {iteration} with confidence {confidence:.3f}")
                break
            
            # 生成下一步query (简化版)
            # TODO: 实现完整的query生成逻辑
            print(f"Iteration {iteration}: confidence={confidence:.3f}")
        
        # 最终结果 = 最后一轮heatmap 或 多轮平均
        final_heatmap = current_heatmap
        if len(all_heatmaps) > 1:
            # 多轮加权平均，越近权重越高
            weights = torch.tensor(
                [0.2, 0.3, 0.5][:len(all_heatmaps)],
                device=final_heatmap.device
            ).unsqueeze(-1).unsqueeze(-1)
            final_heatmap = sum(h['heatmap'] * w for h, w in zip(all_heatmaps, weights))
        
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        
        return {
            'heatmap': final_heatmap,
            'confidence': avg_confidence,
            'iterations': len(all_heatmaps),
            'all_iterations': all_heatmaps
        }


def demo_usage():
    """演示CAILA的基本用法"""
    from PIL import Image
    
    # 配置
    config = CAILAConfig(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        max_iterations=3,
        confidence_threshold=0.85
    )
    
    # 创建模型
    model = CAILA(config)
    
    # 加载示例图像
    # image = Image.open("path/to/test/image.jpg")
    
    # 执行定位
    # result = model.localize(image)
    # print(f"Confidence: {result['confidence']:.3f}")
    # print(f"Heatmap shape: {result['heatmap'].shape}")
    
    return model


if __name__ == "__main__":
    # 基本测试
    print("CAILA model structure:")
    config = CAILAConfig()
    model = CAILA(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("CAILA module initialized successfully")
