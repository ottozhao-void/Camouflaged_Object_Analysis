import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import SGD

import tqdm
import wandb

from omegaconf import OmegaConf
from argparse import ArgumentParser

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import PolynomialLR
from evaluate import evaluate, get_vt_dataset, setup_vt_dataloader, resize_labels

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to your config file (e.g., camouflage.yaml)"
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for DistributedDataParallel"
    )
    return parser.parse_args()

def get_params(model, key):
    """
    Retrieve parameter groups for applying different LR to base layers and ASPP in DeepLab.
    """
    if key == "1x":
        for name, module in model.named_modules():
            if "layer" in name and isinstance(module, nn.Conv2d):
                for p in module.parameters():
                    yield p
    elif key == "10x":
        for name, module in model.named_modules():
            if "aspp" in name and isinstance(module, nn.Conv2d):
                yield module.weight
    elif key == "20x":
        for name, module in model.named_modules():
            if "aspp" in name and isinstance(module, nn.Conv2d):
                yield module.bias


def main():

    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    CONFIG = OmegaConf.load(args.config_path)

    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
    
    # Initialize from pretrained if needed
    if CONFIG.MODEL.INIT_MODEL:
        state_dict = torch.load(
            CONFIG.MODEL.INIT_MODEL,
            map_location={"cuda:0": f"cuda:{local_rank}"}
        )

        model.base.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # Define loss
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([0.1, 0.9], device=device)
        ).to(device)

    # Optimizer
    optimizer = SGD(
        [
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

    # Scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.EPOCH,
        power=CONFIG.SOLVER.POLY_POWER
    )

    # --------------------------------------------
    # 4) Build Datasets & DataLoaders
    # --------------------------------------------
    # 4.1) Training dataset
    train_dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True
    )
    []
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        sampler=train_sampler,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    
    val_dataset = get_vt_dataset("val", CONFIG)
    val_loader = setup_vt_dataloader(val_dataset, CONFIG, True)

    # --------------------------------------------
    # 5) WandB init (only on rank=0)
    # --------------------------------------------
    if local_rank == 0:
        wandb.init(
            project="Camouflaged_Object_Analysis",
            config=OmegaConf.to_container(CONFIG)
        )
        wandb.watch(model, log="all", log_freq=5)

    # --------------------------------------------
    # 6) Training Loop
    # --------------------------------------------
    num_epochs = CONFIG.SOLVER.EPOCH

    if local_rank == 0:
        epoch_iter = tqdm.trange(num_epochs, desc="Training")
    else:
        epoch_iter = range(num_epochs)

    for epoch in epoch_iter:
        # Set sampler epoch for DistributedSampler
        train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        total_steps = len(train_loader)

        # progress bar for the training loop
        if local_rank == 0:
            train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}", total=total_steps)
        else:
            train_pbar = train_loader

        # --------------------------
        # Mini-batch iteration
        # --------------------------
        for images, labels in train_pbar:
            images = images.to(device)
            
            # The model returns a list of feature maps for multi-scale outputs
            logits_list = model(images)

            # Sum up losses from each scale output
            loss_total = 0.0
            for logits in logits_list:
                # Adjust labels to match each logits scale
                H, W = logits.shape[-2:]
                labels_resized = resize_labels(labels, (W, H)).to(device, dtype=torch.long)

                loss_total += criterion(logits, labels_resized)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            running_loss += loss_total.item()

        scheduler.step()

        # Average epoch loss across processes
        epoch_loss = torch.tensor(running_loss / total_steps, device=device)
        dist.reduce(epoch_loss, 0, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss / world_size

        # Validate at the end of each epoch
        evaluate(model, val_loader, config=CONFIG, distributed=True)

        # Rank 0 logs
        if local_rank == 0:
            print(f"[Epoch {epoch + 1}/{num_epochs}] => Loss: {epoch_loss.item():.4f}")
            print(f"  => LR: {scheduler.get_lr()}")

            # Log to WandB
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss.item(),
                "lr": scheduler.get_lr()[0] if isinstance(scheduler.get_lr(), list) else scheduler.get_lr()
            })

            if (epoch + 1) % CONFIG.SOLVER.CHECKPOINT == 0:
                 ckpt_path = f"checkpoint_{epoch + 1}.pth"
                 torch.save(model.module.state_dict(), ckpt_path)
                 print(f"Saved checkpoint: {ckpt_path}")

        # Ensure all ranks sync here before next epoch
        dist.barrier()

    # Training completed
    if local_rank == 0:
        print("Training finished!")
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
