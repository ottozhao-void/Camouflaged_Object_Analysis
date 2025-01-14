import json
import os
import joblib
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import DenseCRF, PolynomialLR

import wandb
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split

from libs.utils.metric import SegmentationMetric
from .test import get_vt_dataset

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(torch.from_numpy(np.array(label)))
    new_labels = torch.stack(new_labels, 0)
    return new_labels

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA if available")
    parser.add_argument("--config-path", type=str, required=True, default="/data/sinopec/xjtu/zfh/Advanced_ML_Coursework/configs/camouflage.yaml", help="Path to the configuration file")
    parser.add_argument("--local-rank", type=int, default=0, help="Rank of the process")
    return parser.parse_args()
        
def test(model, val_loader, task):
    """
    在验证逻辑中，进行如下步骤
    1. 使用多进程同时对验证集进行性能度量
    2. 使用`wandb`追踪并可视化这些指标
    3. 根据性能指标决定是否保存断点
    
    """
    
    device = next(model.parameters()).device
    assert task in ["val", "test"], f"任务只能是'val'或者'test'，但是得到了{task}"
    
    model.eval()
    mae = 0
    sm = 0
    if dist.get_rank()==0:
        val_pbar = tqdm(val_loader, desc=f"{task}", total=len(val_loader))
    else:
        val_pbar = val_loader
    with torch.no_grad():
        for images, labels in val_pbar:
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
            pred_labels = torch.argmax(prob, dim=1).to(torch.float32).cpu()

            for pred_label, gt_label in zip(pred_labels, labels):
                mae += SegmentationMetric.calculate_mae(pred_label, gt_label)
                sm += SegmentationMetric.calculate_smeasure(pred_label, gt_label)
        
        mae = mae.to(device)
        sm = sm.to(device)  
        dist.reduce(tensor=mae, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(tensor=sm, dst=0, op=dist.ReduceOp.SUM)
            
        if dist.get_rank()==0:
            num_samples = len(val_loader.dataset)
            mae /= num_samples
            sm /= num_samples
            
            print("\n=====================================")
            print(f"{task} MAE: {mae:.4f}")
            print(f"{task} S-Measure: {sm:.4f}")
            print("=====================================")

def main():
    
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl",  init_method="env://", rank=local_rank)
    
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    
    args = parse_args()
    CONFIG = OmegaConf.load(args.config_path)
    
    # Dataset Setup

    train_loader, val_loader, train_ddp_sampler, test_dataset = setup_data_loaders(local_rank, world_size, CONFIG)
    num_batches = len(train_loader)
    
    ## MODEL SETUP
    print(f"Model: {CONFIG.MODEL.NAME} on {torch.cuda.get_device_name()} {local_rank}🚀")
    
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL, map_location={f'cuda:{0}': f'cuda:{local_rank}'})
    
    if local_rank==0:
        print("    Init:", CONFIG.MODEL.INIT_MODEL)    
        for m in model.base.state_dict().keys():
            if m not in state_dict.keys():
                print("    Skip init:", m)
    
    model.base.load_state_dict(state_dict, strict=False)  # to skip ASPP
    model.to(device)
    model = DDP(model,device_ids=[local_rank])
    
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model, key="1x"),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model, key="10x"),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model, key="20x"),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )

    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )
    
    # Wandb Writter
    if local_rank==0:
        run = wandb.init(
            project="Camouflaged_Object_Analysis",
            config=OmegaConf.to_container(CONFIG)
        )
    

    if local_rank==0:
        epoch_pbar = tqdm(range(CONFIG.SOLVER.EPOCH))
    else:
        epoch_pbar = range(CONFIG.SOLVER.EPOCH)
        
    for epoch in epoch_pbar:
        
        model.train()
        train_ddp_sampler.set_epoch(epoch)
        epoch_loss = 0
        
        if local_rank==0:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", total=num_batches)
        else:
            train_pbar = train_loader
            
        for images, labels in train_pbar:
            images = images.to(device)
            
            logits = model(images)

            _loss = 0
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                H, W = logit.shape[-2:]
                labels_ = resize_labels(labels, size=(H, W))
                _loss += criterion(logit, labels_.to(dtype=torch.long).to(device))
            
            epoch_loss += _loss
            
            _loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()
        
        # Averge the loss across process group 
        epoch_loss /= num_batches
        dist.reduce(tensor=epoch_loss, dst=0, op=dist.ReduceOp.SUM)
        epoch_loss /= world_size
        
        test(model, val_loader, task="val")
        
        # Inter-Epoch Logic
        if local_rank==0:
            
            print(f"Epoch {epoch+1} Loss: {epoch_loss}")
            print(f"Epoch {epoch+1} Learning Rate: {scheduler.get_lr()}")
            print("=====================================")
            # track learning rate)
            # wandb.log({
            #     "epoch": epoch,
            #     "loss": epoch_loss,
            #     "s-measure": metric.get_smeasure(),
            #     "mae": metric.get_mae(),
            #     "lr": scheduler.get_lr()
            # })
            
            # # Save checkpoint at every `CONFIG.SOLVER.CHECKPOINT` epoch
            # if epoch%CONFIG.SOLVER.CHECKPOINT==0:
            #     torch.save(model.state_dict(), f"checkpoint_{epoch}.pth")
            
            # # Save the model with the best s-measure
            # if metric.get_smeasure() > metric.best_smeasure:
            #     metric.best_smeasure = metric.get_smeasure()
            #     torch.save(model.state_dict(), "best_smeasure.pth")
        
        dist.barrier()

    
    
def setup_data_loaders(local_rank, world_size, CONFIG):
    
    train_dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True,
    )
    train_ddp_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=local_rank
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        sampler=train_ddp_sampler
    )
    
    val_dataset = get_vt_dataset("val", CONFIG)
    
    val_ddp_sampler = DistributedSampler(
        dataset=val_dataset,
        num_replicas=world_size,
        rank=local_rank
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.VAL,
        sampler=val_ddp_sampler
    )
    
    return train_loader, val_loader, train_ddp_sampler

if __name__ == "__main__":
    main()
    dist.destroy_process_group()
  