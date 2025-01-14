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
from torch.utils.data import DataLoader, SubsetRandomSampler

from libs.utils.metric import Metric

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

def crf(config_path, n_jobs):
    """
    CRF post-processing on pre-computed logits
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TEST,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        augment=False,
    )
    print(dataset)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )

    # Path to logit files
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TEST,
        "logit",
    )
    print("Logit src:", logit_dir)
    if not os.path.isdir(logit_dir):
        print("Logit not found, run first: python main.py test [OPTIONS]")
        quit()

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "scores",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TEST,
    )
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores_crf.json")
    print("Score dst:", save_path)

    # Process per sample
    def process(i):
        image_id, image, gt_label = dataset.__getitem__(i)

        filename = os.path.join(logit_dir, image_id + ".npy")
        logit = np.load(filename)

        _, H, W = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)
        label = np.argmax(prob, axis=0)

        return label, gt_label

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(dataset))]
    )

    preds, gts = zip(*results)

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA if available")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the configuration file")
    
    return parser.parse_args()
        
def test(model, val_loader, metric, task):
    """
    在验证逻辑中，进行如下步骤
    1. 度量模型`model`在验证集上的性能指标
    2. 使用`wandb`追踪并可视化这些指标
    3. 根据性能指标决定是否保存断点
    
    """
    
    device = next(model.parameters()).device
    assert task in ["val", "test"], f"任务只能是'val'或者'test'，但是得到了{task}"
    
    model.eval()
    metric.reset()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            
            _, H, W = labels.shape
            logits = F.interpolate(
                logits,
                size = (H, W),
                mode="bilinear",
                align_corners=False
            )
            
            prob = torch.softmax(logits, dim=1)
            pred_labels = torch.argmax(prob, dim=1)

            for pred_label, gt_label in zip(pred_labels, labels):
                metric.add(pred_label, gt_label)

def get_val_test_loader(config):
    test_dataset = get_dataset(config.DATASET.NAME)(
        root=config.DATASET.ROOT,
        split=config.DATASET.SPLIT.TEST,
        ignore_label=config.DATASET.IGNORE_LABEL,
        augment=False,
        base_size=config.IMAGE.SIZE.BASE,
        crop_size=config.IMAGE.SIZE.TEST
    )

    indices = list(range(len(test_dataset)-2))
    np.random.seed(config.DATASET.SEED)
    np.random.shuffle(indices)

    val_portion = int(config.DATASET.PORTION.VAL*len(test_dataset))
    val_indices, test_indices = indices[:val_portion], indices[val_portion:]

    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    val_loader = DataLoader(
        test_dataset,
        batch_size=config.SOLVER.BATCH_SIZE.VAL,
        sampler=val_sampler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.SOLVER.BATCH_SIZE.TEST,
        sampler=test_sampler
    )
    
    return val_loader, test_loader

def main():
    
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl",  init_method="env://", rank=local_rank)
    
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    
    args = parse_args()
    CONFIG = OmegaConf.load(args.config_path)
    
    # Dataset Setup
    #TODO: Change Split to train
    train_dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TEST,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True,
    )
    dsp_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=local_rank
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        sampler=dsp_sampler
    )
    
    if local_rank==0:
        val_loader, test_loader = get_val_test_loader(CONFIG)
        
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
    
    model.train()
    num_batches = len(train_loader) // world_size
    
    if local_rank==0:
        metric = Metric(device)
        
    for epoch in tqdm(range(CONFIG.SOLVER.EPOCH)):
        dsp_sampler.set_epoch(epoch)
        
        epoch_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", total=num_batches):
            images = images.to(device)
            
            logits = model(images)
            
            _loss = 0
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                _, _, H, W = logit.shape
                labels_ = resize_labels(labels, size=(H, W))
                _loss += criterion(logit, labels_.to(dtype=torch.long, device=device))
            
            epoch_loss += _loss
            
            _loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()
        
        # Averge the loss across process group 
        epoch_loss /= num_batches
        dist.reduce(tensor=epoch_loss, dst=0, op=dist.ReduceOp.SUM)
        epoch_loss /= world_size
        
        # Inter-Epoch Logic
        if local_rank==0:
            
            test(model, val_loader, metric, task="val")
            wandb.log({
                "epoch": epoch,
                "loss": epoch_loss,
                "s-measure": metric.get_smeasure(),
                "mae": metric.get_mae()
            })
            
            # Save checkpoint at every `CONFIG.SOLVER.CHECKPOINT` epoch
            if epoch%CONFIG.SOLVER.CHECKPOINT==0:
                torch.save(model.state_dict(), f"checkpoint_{epoch}.pth")
            
            # Save the model with the best s-measure
            if metric.get_smeasure() > metric.best_smeasure:
                metric.best_smeasure = metric.get_smeasure()
                torch.save(model.state_dict(), "best_smeasure.pth")
        
        dist.barrier()

if __name__ == "__main__":
    main()
    dist.destroy_process_group()
  