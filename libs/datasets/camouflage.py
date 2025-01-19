import torch.utils.data as data
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as t
import torch
import tqdm
from torch.utils.data import DataLoader
from .base import _BaseDataset
import numpy as np
import random

class CamouflageDataset(_BaseDataset):
    def __init__(self, split_ratio, dataset="none", **kargs):
        self.split_ratio = split_ratio
        
        assert dataset in ["COD10K", "CAMO", "CAMO+COD10K","COD10K+CAMO"], "Dataset not found"
        self.dataset = dataset
        self.image_list = []
        self.mask_list = []
        super().__init__(**kargs)

    def _load_data(self, index):
        # Image preprocessing:
        image = np.asarray(Image.open(self.image_list[index]))
        mask = np.asarray(Image.open(self.mask_list[index]))
        
        # print(f"Processing image: {self.image_list[index]}")
        # print(f"Processing mask: {self.mask_list[index]}")
        
        return image, mask
    
    def _set_files(self):
        self.split_dir = os.path.join(self.root, self.split)
        
        imageP = Path(os.path.join(self.split_dir, "image"))
        maskP = Path(os.path.join(self.split_dir, "GT"))
        
        if self.split == "test":
            for d in self.dataset.split("+"):
                selected_images, selected_masks = self.split_dataset(imageP, maskP, "camourflage" if d == "CAMO" else d)
                self.image_list.extend(selected_images)
                self.mask_list.extend(selected_masks)
        else:
            self.image_list = sorted([img for img in imageP.iterdir()])
            self.mask_list = sorted([mask for mask in maskP.iterdir()])

        return
    def split_dataset(self, imageP, maskP, dataset):
        random.seed(42)
        
        assert self.split_ratio is not None, "Split ratio is not defined"
        image_dir = imageP.resolve()
        mask_dir = maskP.resolve()
        
        image_list = []
        mask_list = []

        for image in imageP.iterdir():
            if image.name.startswith(dataset):
                image_list.append(os.path.join(image_dir, image.name))
                mask_list.append(os.path.join(mask_dir, image.name.replace("jpg", "png")))
            
        num_to_select = int(len(image_list) * self.split_ratio)
        selcted_indexes = random.sample(range(len(image_list)), num_to_select)
        selected_images = [image_list[i] for i in selcted_indexes]
        selected_masks = [mask_list[i] for i in selcted_indexes]
        
        return selected_images, selected_masks

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
