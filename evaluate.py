import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split

import numpy as np
import tqdm
import wandb

from libs.utils.metric import SegmentationMetric
from libs.datasets import get_dataset
from libs.utils import DenseCRF
from libs.models import DeepLabV2_ResNet101_MSC


from omegaconf import OmegaConf

def resize_labels(labels, size):
    """
    Resize label tensors to match the shape (H, W) of the logits at each scale.
    Uses nearest neighbor interpolation.
    """
    from PIL import Image
    import numpy as np

    new_labels = []
    for label in labels:
        label_np = label.float().numpy()
        label_img = Image.fromarray(label_np).resize(size, resample=Image.NEAREST)
        new_labels.append(torch.from_numpy(np.array(label_img)))
    return torch.stack(new_labels, dim=0)

def setup_vt_dataloader(dataset, config, distributed=False):
    """
    1) 配置用于验证/测试阶段的数据加载器
    2) 如果分布式，则需要使用 DistributedSampler 以确保各进程拿到不同的子集
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

    # 根据是 test 还是 val，使用不同的 batch size
    if dataset.task == "test":
        bsize = config.SOLVER.BATCH_SIZE.TEST
    else:
        bsize = config.SOLVER.BATCH_SIZE.VAL

    data_loader = DataLoader(
        dataset,
        batch_size=bsize,
        sampler=sampler,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=True
    )
    return data_loader


def crf_refine(image_batch, prob_batch, config):
    """
    在推理时对分割概率图做 CRF 后处理 (batch-wise).
    假设:
        image_batch: (N, 3, H, W)
        prob_batch: (N, C, H, W)
    返回: refined_prob_batch (N, C, H, W)
    """
    refined_list = []

    # Iterate each sample in the batch
    for img, prob in zip(image_batch, prob_batch):
        # 转成 numpy 格式 (H, W, 3) & (C, H, W)
        # 注意这里的 squeeze/unsqueeze 只对单图有效
        # 对 batch 处理时需要手动分离
        img_np = img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        prob_np = prob.cpu().numpy().astype(np.float32)

        postprocessor = DenseCRF(
            iter_max=config.CRF.ITER_MAX,
            pos_xy_std=config.CRF.POS_XY_STD,
            pos_w=config.CRF.POS_W,
            bi_xy_std=config.CRF.BI_XY_STD,
            bi_rgb_std=config.CRF.BI_RGB_STD,
            bi_w=config.CRF.BI_W,
        )
        refined_prob_np = postprocessor(img_np, prob_np)  # shape: (C, H, W)

        # 转回 torch 张量
        refined_prob_t = torch.from_numpy(refined_prob_np).to(prob.device)
        refined_list.append(refined_prob_t)

    # stack 回成 batch
    refined_prob_batch = torch.stack(refined_list, dim=0)  # (N, C, H, W)
    return refined_prob_batch


def convert_image_pred_gt(image, pred, gt):
    """
    将三者转换为适合 wandb.Image() 可视化的 numpy 形式:
      - image (3,H,W) in [0,1] -> (H,W,3) in [0,255], dtype=uint8
      - pred (H,W) in {0,1} -> (H,W) in {0,255}, dtype=uint8
      - gt   (H,W) in {0,1} -> (H,W) in {0,255}, dtype=uint8
    """
    # image: (3, H, W)
    image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # pred & gt: (H, W)
    pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)
    gt_np   = (gt.cpu().numpy()   * 255).astype(np.uint8)

    return image_np, pred_np, gt_np


def log_mask(image, pred, gt):
    """
    将图像及掩码通过 wandb.log() 进行可视化:
       - image: (3,H,W) in [0,1], torch.float32
       - pred:  (H,W) in {0,1},  torch.int64/float32
       - gt:    (H,W) in {0,1},  torch.int64/float32
    """
    image_np, pred_np, gt_np = convert_image_pred_gt(image, pred, gt)

    masked_image = wandb.Image(
        image_np,
        masks={
            "prediction": {"mask_data": pred_np, "class_labels": {0: "background", 255: "camouflage"}},
            "ground_truth": {"mask_data": gt_np, "class_labels": {0: "background", 255: "camouflage"}}
        }
    )
    wandb.log({"mask": masked_image})


def get_vt_dataset(task, config):
    """
    依据 task = "val" 或 "test" 构造 dataset。
    这里按照 config.DATASET.<...> 的参数进行构造后，
    再用 random_split 按照指定比例拆分出 val/test 数据集。
    """
    assert task in ["val", "test"], f"任务只能是'val'或者'test'，但是得到了 {task}"

    dataset_all = get_dataset(config.DATASET.NAME)(
        root=config.DATASET.ROOT,
        split=config.DATASET.SPLIT.TEST,
        augment=False,
        base_size=config.IMAGE.SIZE.BASE,
        crop_size=config.IMAGE.SIZE.TEST,
        task=task
    )

    # 按一定比例拆分数据集
    split_ratio = config.DATASET.TEST_SIZE if task == "test" else config.DATASET.VAL_SIZE

    dataset_subset, _ = random_split(
        dataset_all,
        [split_ratio, 1 - split_ratio],
        generator=torch.Generator().manual_seed(config.DATASET.SEED)
    )
    # 这里给 dataset_subset 附个小标记，以便在 setup_vt_dataloader 中区分
    dataset_subset.task = task

    return dataset_subset


def evaluate(model, data_loader, config, distributed=False, compute_loss=False):
    """
    统一的验证/测试逻辑:
      1. 根据 task 构造数据集 + DataLoader
      2. 推断 (model forward)
      3. (test 时) 使用 CRF 后处理
      4. 计算并记录指标 (MAE / S-measure)
      5. 可选地可视化若干张图
      6. 如为分布式，多进程同步累加结果
      7. 只在 rank=0 进程上进行 wandb.log
    """
    device = next(model.parameters()).device
    task = data_loader.dataset.task
    assert task in ["val", "test"], f"任务只能是'val'或者'test'，但是得到了 {task}"

    # 用于累加所有样本的评估指标
    mae = 0.0
    sm  = 0.0

    # 只在 test 时可视化若干张结果
    num_visualize = config.DATASET.NUM_VISUALIZE if task == "test" else 0

    # 构建进度条 (只在 rank=0 进程可见)
    if (distributed and dist.get_rank() == 0) or not distributed:
        progress_bar = tqdm.tqdm(data_loader, desc=f"{task}", total=len(data_loader))
    else:
        progress_bar = data_loader

    if compute_loss:
        criterion = nn.CrossEntropyLoss().to(device)
        val_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for batch in progress_bar:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)  # (N, H, W)

            # ---- Forward ----
            logits = model(images)  # (N, C, H', W')
            
            if compute_loss:
                H, W = logits.shape[-2:]
                labels_resized = resize_labels(labels.cpu(), (W, H)).to(device, dtype=torch.long)

                val_loss += criterion(logits, labels_resized)

            # 调整大小到 (H, W) 以跟 labels 对齐
            _, H, W = labels.shape
            logits = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

            # ---- 得到概率图 ----
            prob = torch.softmax(logits, dim=1)  # (N, C, H, W)

            # # ---- 如果是 test, 进行 CRF 后处理 ----
            # if task == "test":
            #     prob = crf_refine(images, prob, config)  # (N, C, H, W)

            # ---- 得到预测分割 (N, H, W) ----
            pred_labels = torch.argmax(prob, dim=1).float().cpu()

            # ---- 计算指标 & 可选可视化 ----
            for i in range(pred_labels.size(0)):
                # 单张预测 & 标签
                pred_label = pred_labels[i]
                gt_label   = labels[i].cpu()

                # 统计
                mae += SegmentationMetric.calculate_mae(pred_label, gt_label)
                sm  += SegmentationMetric.calculate_smeasure(pred_label, gt_label)

                # 仅在 test 时可视化若干图像
                if num_visualize > 0 and wandb.run is not None:
                    # 注意: images[i] 是 (3,H,W), 在 [0,1] 范围
                    log_mask(images[i], pred_label, gt_label)
                    num_visualize -= 1

    # 3) 分布式同步
    mae = mae.to(device)
    sm  = sm.to(device)

    if distributed:
        dist.reduce(mae, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(sm,  dst=0, op=dist.ReduceOp.SUM)

    # 4) rank=0 进程上统一记录
    if (not distributed) or (dist.get_rank() == 0):
        num_samples = len(data_loader.dataset)
        mae = mae / num_samples
        sm  = sm  / num_samples
        
        if wandb.run is not None:
            wandb.log({f"{task}_mae": mae.item()})
            wandb.log({f"{task}_smeasure": sm.item()})

        print(f"[{task.upper()}] => MAE: {mae.item():.4f}, S-Measure: {sm.item():.4f}")
        
        if compute_loss:
            val_loss = val_loss / num_samples
            print(f"[{task.upper()}] => Loss: {val_loss.item():.4f}")

if __name__ == "__main__":
    
    run = wandb.init(project="Camouflaged_Object_Analysis", tags=["test"])
    config = OmegaConf.load("/data/sinopec/xjtu/zfh/Advanced_ML_Coursework/configs/camouflage.yaml")
    device = torch.device("cuda:1")
    # Load model
    model = DeepLabV2_ResNet101_MSC(n_classes=config.DATASET.N_CLASSES).to(device)
    state_dict = torch.load("/data/sinopec/xjtu/zfh/Advanced_ML_Coursework/checkpoint_20.pth")
    model.load_state_dict(state_dict)
    
    test_dataset = get_vt_dataset("test", config)
    test_loader = setup_vt_dataloader(test_dataset, config, distributed=False)
    
    evaluate(model, test_loader, config, distributed=False, compute_loss=True)