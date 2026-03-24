"""
数据集下载脚本
使用HuggingFace镜像加速下载
"""

import os
import argparse
import logging
from pathlib import Path

# 设置HF镜像
HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HF_ENDPOINT'] = HF_ENDPOINT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_dataset(dataset_name: str, output_dir: str):
    """下载HuggingFace数据集"""
    try:
        from huggingface_hub import snapshot_download
        
        logger.info(f"Downloading dataset: {dataset_name}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Using HF_ENDPOINT: {HF_ENDPOINT}")
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 下载数据集
        local_dir = snapshot_download(
            repo_type='dataset',
            repo_id=dataset_name,
            local_dir=output_dir,
            resume_download=True
        )
        
        logger.info(f"Dataset downloaded successfully to: {local_dir}")
        return True
        
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def prepare_inpainting_data(output_dir: str):
    """准备Inpainting数据集结构"""
    data_root = Path(output_dir)
    
    for split in ['train', 'val', 'test']:
        (data_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (data_root / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created data directory structure at: {data_root}")


def main():
    parser = argparse.ArgumentParser(description='Download datasets for CAILA')
    parser.add_argument('--dataset', type=str, default='facebook/partial-scribbles-dataset',
                       help='Dataset name on HuggingFace')
    parser.add_argument('--output', type=str, default='.',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 准备目录结构
    prepare_inpainting_data(args.output)
    
    # 下载数据集
    success = download_dataset(args.dataset, args.output)
    
    if success:
        logger.info("Data preparation complete!")
    else:
        logger.warning("Data download failed. Using synthetic test data for development.")


if __name__ == '__main__':
    main()
