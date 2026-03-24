"""
CAILA训练脚本
支持DDP、混合精度、WandB日志
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import wandb

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CAILA model')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--decoder_output_size', type=int, default=14)
    parser.add_argument('--max_iterations', type=int, default=3)
    
    # 分布式配置
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    
    # 其他配置
    parser.add_argument('--output_dir', type=str, default='../05_runs')
    parser.add_argument('--data_dir', type=str, default='../03_data')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='CAILA')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def setup_distributed(args):
    """设置分布式训练"""
    if args.world_size > 1:
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        
        return True
    return False


def cleanup_distributed():
    """清理分布式训练"""
    destroy_process_group()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class InpaintingDataset(Dataset):
    """Inpainting篡改定位数据集"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: int = 448
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        
        # 加载样本索引
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []
        
        # 从数据目录加载
        image_dir = self.data_dir / self.split / 'images'
        mask_dir = self.data_dir / self.split / 'masks'
        
        if image_dir.exists():
            for img_path in sorted(image_dir.glob('*.png')):
                img_id = img_path.stem
                mask_path = mask_dir / f'{img_id}.png'
                
                if mask_path.exists():
                    samples.append({
                        'image': str(img_path),
                        'mask': str(mask_path)
                    })
        
        # 如果没有数据，返回空列表
        if len(samples) == 0:
            logger.warning(f"No samples found in {image_dir}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        from PIL import Image
        
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image']).convert('RGB')
        mask = Image.open(sample['mask']).convert('L')
        
        # 调整大小
        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size))
        
        # 转换为tensor
        image_tensor = torch.from_numpy(
            np.array(image).transpose(2, 0, 1)
        ).float() / 255.0
        
        mask_tensor = torch.from_numpy(
            np.array(mask)
        ).float() / 255.0
        
        # 下采样mask到decoder输出尺寸
        output_size = 14  # TODO: 从config读取
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(output_size, output_size),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'image_path': sample['image']
        }


class CAILALoss(nn.Module):
    """CAILA专用损失函数"""
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 0.5,
        iou_weight: float = 0.5
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
    
    def bce_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(pred, target)
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        return 1 - (2 * intersection + smooth) / (union + smooth)
    
    def iou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        return 1 - (intersection + smooth) / (union + smooth)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        iou = self.iou_loss(pred, target)
        
        total = (
            self.bce_weight * bce +
            self.dice_weight * dice +
            self.iou_weight * iou
        )
        
        return {
            'total': total,
            'bce': bce,
            'dice': dice,
            'iou': iou
        }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CAILALoss,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_iou = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 混合精度前向传播
        with autocast():
            # TODO: 实现完整的前向传播
            # outputs = model(images)
            # losses = criterion(outputs['heatmap'], masks)
            
            # 临时：使用随机输出进行测试
            batch_size = images.shape[0]
            outputs = torch.rand(batch_size, 1, 14, 14).to(device)
            losses = criterion(outputs, masks)
        
        loss = losses['total'] / args.gradient_accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 记录损失
        total_loss += losses['total'].item()
        total_bce += losses['bce'].item()
        total_dice += losses['dice'].item()
        total_iou += losses['iou'].item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {losses['total'].item():.4f} | "
                f"BCE: {losses['bce'].item():.4f} | "
                f"Dice: {losses['dice'].item():.4f}"
            )
            
            if args.use_wandb:
                wandb.log({
                    'train/batch_loss': losses['total'].item(),
                    'train/batch_bce': losses['bce'].item(),
                    'train/batch_dice': losses['dice'].item(),
                    'epoch': epoch,
                    'batch': batch_idx
                })
    
    return {
        'loss': total_loss / num_batches,
        'bce': total_bce / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches
    }


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: CAILALoss,
    device: torch.device
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    
    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 临时：使用随机输出进行测试
            batch_size = images.shape[0]
            outputs = torch.rand(batch_size, 1, 14, 14).to(device)
            losses = criterion(outputs, masks)
            
            total_loss += losses['total'].item()
            total_bce += losses['bce'].item()
            total_dice += losses['dice'].item()
            total_iou += losses['iou'].item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'bce': total_bce / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    args,
    is_best: bool = False
):
    """保存检查点"""
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'args': vars(args)
    }
    
    # 保存最新检查点
    latest_path = checkpoint_dir / 'latest.pt'
    torch.save(checkpoint, latest_path)
    
    # 保存最佳检查点
    if is_best:
        best_path = checkpoint_dir / 'best.pt'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint with metric: {best_metric:.4f}")
    
    # 保存epoch检查点
    epoch_path = checkpoint_dir / f'epoch_{epoch}.pt'
    torch.save(checkpoint, epoch_path)


def main():
    args = parse_args()
    
    # 设置分布式
    is_distributed = setup_distributed(args)
    
    # 设置设备
    device = torch.device(
        f'cuda:{args.local_rank}' if torch.cuda.is_available() and args.local_rank >= 0 else 'cpu'
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化WandB
    if args.use_wandb and args.rank == 0:
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
    
    # 加载数据
    train_dataset = InpaintingDataset(
        data_dir=args.data_dir,
        split='train'
    )
    
    val_dataset = InpaintingDataset(
        data_dir=args.data_dir,
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 加载模型
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import CAILA, CAILAConfig
    
    model_config = CAILAConfig(
        model_name=args.model_name,
        max_iterations=args.max_iterations
    )
    model = CAILA(model_config)
    
    # 将decoder移到设备（MLLM延迟加载）
    model.decoder = model.decoder.to(device)
    model = model.to(device)
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # 混合精度scaler
    scaler = GradScaler()
    
    # 损失函数
    criterion = CAILALoss()
    
    # 恢复检查点
    start_epoch = 0
    best_metric = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_metric']
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # 训练循环
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, args
        )
        
        # 评估
        if len(val_loader) > 0:
            val_metrics = evaluate(model, val_loader, criterion, device)
        else:
            val_metrics = train_metrics
        
        # 记录指标
        if args.rank == 0:
            logger.info(
                f"Epoch {epoch} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Dice: {val_metrics['dice']:.4f}"
            )
            
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/bce': train_metrics['bce'],
                    'train/dice': train_metrics['dice'],
                    'train/iou': train_metrics['iou'],
                    'val/loss': val_metrics['loss'],
                    'val/bce': val_metrics['bce'],
                    'val/dice': val_metrics['dice'],
                    'val/iou': val_metrics['iou']
                })
            
            # 保存检查点
            is_best = val_metrics['dice'] < best_metric
            if is_best:
                best_metric = val_metrics['dice']
            
            save_checkpoint(model, optimizer, epoch, best_metric, args, is_best)
    
    # 清理分布式
    if is_distributed:
        cleanup_distributed()
    
    if args.use_wandb:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
