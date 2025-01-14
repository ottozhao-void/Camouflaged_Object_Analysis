import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split

import tqdm
from libs.utils.metric import SegmentationMetric
from libs.datasets import get_dataset
from libs.utils import DenseCRF

import numpy as np
import wandb

def setup_vt_dataloader(dataset, config, distributed=False):
    """
    设置验证/测试数据加载器
    """
        
    if distributed and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
    else:
        sampler = None
    
    bsize = config.SOLVER.BATCH_SIZE.TEST if dataset.task == "test" else config.SOLVER.BATCH_SIZE.VAL
    
    data_loader = DataLoader(
        dataset,
        batch_size=bsize,
        sampler=sampler,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=True
    )
    
    return data_loader

def crf_refine(image, prob, config):
    image = image.cpu().numpy().squeeze(0).transpose(1,2,0).astype(np.uint8)
    prob = prob.squeeze(0).cpu().numpy().astype(np.float32)
    
    postprocessor = DenseCRF(
        iter_max=config.CRF.ITER_MAX,
        pos_xy_std=config.CRF.POS_XY_STD,
        pos_w=config.CRF.POS_W,
        bi_xy_std=config.CRF.BI_XY_STD,
        bi_rgb_std=config.CRF.BI_RGB_STD,
        bi_w=config.CRF.BI_W,
    )
    # 3) Apply CRF
    refined_prob_np = postprocessor(image, prob)  # shape: (C, H, W)

    # 4) Convert back to torch.Tensor, restore the batch dimension
    refined_prob = torch.from_numpy(refined_prob_np).unsqueeze(0).to(prob.device)  # (1, C, H, W)

    return refined_prob

def convert_image_mask(image, mask):
    """
    image 是一个(1,3,H,W)，值位于0到1之间的张量，类型为torch.float32
    mask 是一个(H,W)，值位于0到1之间的张量，类型为torch.int64
    
    需要将image变成(H,W,3)的numpy数组，值位于0到255之间，类型为np.uint8
    需要将mask变成(H,W)的numpy数组，值位于0到255之间，类型为np.uint8
    """
    # 检查
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.int64
    assert image.shape[0] == 3
    assert len(mask.shape) == 2
    assert image.shape[1] == mask.shape[0]
    assert image.shape[2] == mask.shape[1]
    assert image.min() >= 0
    assert image.max() <= 1
    assert mask.min() >= 0
    assert mask.max() <= 1
    
    # Convert image to numpy and transpose
    image = (image.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # Convert mask to numpy
    mask = (mask.numpy() * 255).astype(np.uint8)
    return image, mask


def test(model, task, config, distributed=False):
    """
    在验证逻辑中，进行如下步骤
    1. 使用多进程同时对验证集进行性能度量
    2. 使用`wandb`追踪并可视化这些指标, 可视化mask
    3. 根据性能指标决定是否保存断点
    
    """
    
    device = next(model.parameters()).device
    assert task in ["val", "test"], f"任务只能是'val'或者'test'，但是得到了{task}"
    
    dataset = get_vt_dataset(task, config)
    data_loader = setup_vt_dataloader(dataset, config, distributed)
    
    model.eval()
    mae = 0
    sm = 0
    num_visulize = config.DATASET.NUM_VISUALIZE if task == "test" else 0
    if distributed and dist.get_rank()==0:
        pabr = tqdm(data_loader, desc=f"{task}", total=len(data_loader))
    else:
        pabr = data_loader
        
    with torch.no_grad():
        for images, labels in pabr:
            images = images.to(device)
            labels = labels.to(torch.float32)
            
            logits = model(images)
            
            _, H, W = labels.shape
            logits = F.interpolate(
                logits,
                size = (H, W),
                mode="bilinear",
                align_corners=False
            )
            
            prob = torch.softmax(logits, dim=1)
            
            if task == "test":
                prob = crf_refine(images, prob)
            
            pred_labels = torch.argmax(prob, dim=1).to(torch.float32).cpu()

            # Metric calculation and visualization
            for pred_label, gt_label in zip(pred_labels, labels):
                mae += SegmentationMetric.calculate_mae(pred_label, gt_label)
                sm += SegmentationMetric.calculate_smeasure(pred_label, gt_label)
                
                if num_visulize > 0:
                    log_mask(images.cpu(), pred_label, gt_label)
                    num_visulize -= 1
        
        mae = mae.to(device)
        sm = sm.to(device)  

        if distributed:
            dist.reduce(tensor=mae, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(tensor=sm, dst=0, op=dist.ReduceOp.SUM)
            
            if dist.get_rank()==0:
                num_samples = len(data_loader.dataset)
                mae /= num_samples
                sm /= num_samples
            
                wandb.log({f"{task}_mae": mae.item()})
                wandb.log({f"{task}_smeasure": sm.item()})
        else:
            num_samples = len(data_loader.dataset)
            mae /= num_samples
            sm /= num_samples
            wandb.log({f"{task}_mae": mae.item()})
            wandb.log({f"{task}_smeasure": sm.item()})
            
            
def get_vt_dataset(task, config):
    
    assert task in ["val", "test"], f"任务只能是'val'或者'test'，但是得到了{task}"
    
    dataset = get_dataset(config.DATASET.NAME)(
        root=config.DATASET.ROOT,
        split=config.DATASET.SPLIT.TEST,
        ignore_label=config.DATASET.IGNORE_LABEL,
        augment=False,
        base_size=config.IMAGE.SIZE.BASE,
        crop_size=config.IMAGE.SIZE.TEST,
        task=task
    )
    
    split_ratio = config.DATASET.TEST_SIZE if task == "test" else config.DATASET.VAL_SIZE

    dataset, _ = random_split(
        dataset,
        [split_ratio, 1-split_ratio],
        generator=torch.Generator().manual_seed(config.DATASET.SEED)
    )
    
    return dataset

def log_mask(image, pred, gt):
    image, pred, gt = convert_image_mask(image, pred, gt)
    masked_image = wandb.Image(
        image,
        mask={
            "prediction": {"mask_data": pred},
            "groud_truth": {"mask_data": gt}
        }
    )
    wandb.log({"mask": masked_image})