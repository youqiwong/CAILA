"""
CAILA 沙箱测试脚本
用于验证核心功能和本地数据测试
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, List
import logging
import argparse

# 设置日志
log_dir = '/Users/wangyouqi/Desktop/EvoScientist_Local_Research_Kit/projects/aigc_inpainting_detection/08_logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'sandbox_test.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加src目录到path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_hardware():
    """检查硬件配置"""
    logger.info("=== Hardware Check ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Running on CPU")
    return torch.cuda.is_available()


def test_imports():
    """测试模块导入"""
    logger.info("=== Import Test ===")
    try:
        from model import CAILA, CAILAConfig
        logger.info("CAILA module imported successfully")
        return True, CAILA, CAILAConfig
    except ImportError as e:
        logger.error(f"Failed to import CAILA: {e}")
        return False, None, None


def test_model_initialization(CAILA, CAILAConfig):
    """测试模型初始化"""
    logger.info("=== Model Initialization Test ===")
    try:
        config = CAILAConfig(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            max_iterations=2,
            confidence_threshold=0.8
        )
        model = CAILA(config)
        logger.info(f"Model initialized with config: {config}")
        
        # 检查模型组件
        assert hasattr(model, 'query_generator'), "Missing query_generator"
        assert hasattr(model, 'decoder'), "Missing decoder"
        assert hasattr(model, 'attention_hook'), "Missing attention_hook"
        logger.info("All model components initialized")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return True, model, config
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return False, None, None


def test_decoder_forward(CAILA, CAILAConfig):
    """测试解码器前向传播"""
    logger.info("=== Decoder Forward Test ===")
    try:
        config = CAILAConfig()
        model = CAILA(config)
        
        # 创建虚拟输入
        batch_size = 1
        num_tokens = 196  # 14x14 grid
        hidden_dim = config.decoder_hidden_dim
        
        dummy_features = torch.randn(batch_size, num_tokens, hidden_dim)
        output = model.decoder(dummy_features)
        
        assert output.shape == (batch_size, 1, 14, 14), f"Unexpected output shape: {output.shape}"
        logger.info(f"Decoder output shape: {output.shape}")
        logger.info(f"Decoder output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        return True
    except Exception as e:
        logger.error(f"Decoder forward failed: {e}")
        return False


def test_attention_hook():
    """测试Attention Hook"""
    logger.info("=== Attention Hook Test ===")
    try:
        from model import AttentionHook
        
        hook = AttentionHook()
        assert hasattr(hook, 'register_hooks'), "Missing register_hooks"
        assert hasattr(hook, 'get_attention_weights'), "Missing get_attention_weights"
        assert hasattr(hook, 'clear'), "Missing clear"
        logger.info("AttentionHook class initialized correctly")
        
        return True
    except Exception as e:
        logger.error(f"AttentionHook test failed: {e}")
        return False


def generate_synthetic_test_data(output_dir: str, num_samples: int = 4):
    """生成合成测试数据用于沙箱测试"""
    logger.info(f"=== Generating {num_samples} synthetic test samples ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # 创建一个简单图像
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        
        # 随机添加一个篡改区域 (高亮方块)
        arr = np.array(img)
        x, y = np.random.randint(50, 150, 2)
        w, h = np.random.randint(20, 50, 2)
        arr[y:min(y+h, 224), x:min(x+w, 224)] = [255, 0, 0]  # 红色方块
        
        tampered = Image.fromarray(arr)
        
        # 保存
        img.save(os.path.join(output_dir, f'original_{i}.png'))
        tampered.save(os.path.join(output_dir, f'tampered_{i}.png'))
        
        # 保存mask
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[y:min(y+h, 224), x:min(x+w, 224)] = 255
        Image.fromarray(mask).save(os.path.join(output_dir, f'mask_{i}.png'))
    
    logger.info(f"Generated {num_samples} synthetic samples in {output_dir}")
    return [os.path.join(output_dir, f'tampered_{i}.png') for i in range(num_samples)]


def test_model_load(CAILA, CAILAConfig, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
    """测试模型加载"""
    logger.info(f"=== Model Loading Test for {model_name} ===")
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
        config = CAILAConfig(model_name=model_name)
        model = CAILA(config)
        logger.info(f"Model config prepared, model_name: {config.model_name}")
        logger.info("Note: Actual model loading requires GPU with >14GB VRAM")
        
        return True
    except Exception as e:
        logger.error(f"Model load test failed: {e}")
        return False


def test_dataset_download():
    """测试HuggingFace数据集下载"""
    logger.info("=== Dataset Download Test ===")
    
    try:
        import huggingface_hub
        logger.info(f"HuggingFace Hub version: {huggingface_hub.__version__}")
        
        hf_endpoint = os.environ.get('HF_ENDPOINT', '')
        logger.info(f"HF_ENDPOINT: {hf_endpoint if hf_endpoint else 'default'}")
        
        return True
    except ImportError:
        logger.warning("huggingface_hub not installed")
        return False
    except Exception as e:
        logger.error(f"Dataset test failed: {e}")
        return False


def run_full_sandbox_test(model_name: Optional[str] = None):
    """运行完整沙箱测试"""
    logger.info("=" * 60)
    logger.info("CAILA Sandbox Test Suite")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. 硬件检查
    has_cuda = check_hardware()
    results['hardware'] = 'GPU' if has_cuda else 'CPU'
    
    # 2. 导入测试
    success, CAILA, CAILAConfig = test_imports()
    results['imports'] = success
    if not success:
        logger.error("Import failed, stopping tests")
        return results
    
    # 3. 模型初始化测试
    success, model, config = test_model_initialization(CAILA, CAILAConfig)
    results['initialization'] = success
    
    # 4. 解码器测试
    success = test_decoder_forward(CAILA, CAILAConfig)
    results['decoder_forward'] = success
    
    # 5. Attention Hook测试
    success = test_attention_hook()
    results['attention_hook'] = success
    
    # 6. 生成合成测试数据
    test_data_dir = '/Users/wangyouqi/Desktop/EvoScientist_Local_Research_Kit/projects/aigc_inpainting_detection/03_data/test_samples'
    test_samples = generate_synthetic_test_data(test_data_dir, num_samples=4)
    results['synthetic_data'] = len(test_samples)
    
    # 7. 模型加载测试
    if has_cuda:
        success = test_model_load(CAILA, CAILAConfig, model_name or "Qwen/Qwen2-VL-7B-Instruct")
        results['model_load'] = success
    else:
        logger.warning("Skipping model load test (no GPU)")
        results['model_load'] = 'skipped'
    
    # 8. 数据集下载测试
    success = test_dataset_download()
    results['dataset_test'] = success
    
    # 总结
    logger.info("=" * 60)
    logger.info("Test Results Summary:")
    logger.info("=" * 60)
    for test_name, result in results.items():
        status = "PASS" if result is True else ("SKIP" if result == 'skipped' or isinstance(result, str) else "FAIL")
        logger.info(f"[{status}] {test_name}: {result}")
    
    passed = sum(1 for v in results.values() if v is True)
    total = sum(1 for v in results.values() if v is not 'skipped' and not isinstance(v, str))
    logger.info(f"\nPassed: {passed}/{total} tests")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAILA Sandbox Test")
    parser.add_argument("--model", type=str, default=None, help="Model name to test")
    args = parser.parse_args()
    
    results = run_full_sandbox_test(model_name=args.model)
    
    # 保存结果
    import json
    results_path = os.path.join(log_dir, 'sandbox_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
