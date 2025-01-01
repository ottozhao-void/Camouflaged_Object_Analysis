from torch.utils.data import DataLoader
from Camouflage_Dataset import CamouflageDataset
from tqdm import tqdm
import torch
from pathlib import Path

datset_stats = {
    "rgb512": [[0.4562, 0.4361, 0.3434], [0.1963, 0.1894, 0.1788]]
}

def count_trainImg_stats(dataset_dir, transform, split="train", bsz=128):

    dataset = CamouflageDataset(
        dataset_dir=dataset_dir,
        split=split,
        transform=transform
    )

    train_loader = DataLoader(dataset=dataset, batch_size=bsz, collate_fn=collate_fn)

    mean = 0.0
    std = 0.0
    num_samples = 0

    for images, _ in tqdm(train_loader, desc="Processing Data"):
        images = images.to("cuda:0")
        ns = images.shape[0]
        #TODO: To implement this so that it counts for other colour space
        images = images.view(ns, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += ns
    
    mean /= num_samples
    std /= num_samples
    print(f"Counted {num_samples}/{len(dataset)} samples")
    print(f"Mean: {mean}")
    print(f"Std: {std}")



def collate_fn(data):
    images = []
    masks = []
    for da in data:
        images.append(da[0])
        masks.append(da[1])
    
    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)

    return images, masks

