import torch.utils.data as data
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as t
import torch
import tqdm
from torch.utils.data import DataLoader
from libs.datasets.base import _BaseDataset
import numpy as np

class CamouflageDataset(_BaseDataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def _load_data(self, index):
        # Image preprocessing:
        image = np.asarray(Image.open(self.image_list[index]))
        mask = np.asarray(Image.open(self.mask_list[index]))

        return image, mask
    
    def _set_files(self):
        self.split_dir = os.path.join(self.root, self.split)
        
        image_dir = Path(os.path.join(self.split_dir, "image"))
        mask_dir = Path(os.path.join(self.split_dir, "GT"))

        self.image_list = sorted([img for img in image_dir.iterdir()])
        self.mask_list = sorted([mask for mask in mask_dir.iterdir()])

        return
        

# class CamouflageDataset(data.Dataset):
#     def __init__(self, dataset_dir, split, img_size=(512,512), color_space = "RGB"):
#         super().__init__()

#         self.dataset_dir = dataset_dir

#         self.imgsz = img_size
#         self.color_space = color_space
#         self.stats_mean, self.stats_std = datset_stats[f"{color_space.lower()}{img_size[0]}"]

#         # Preprocessing
#         self.mask_transform = t.Compose([
#             t.Resize(img_size),
#             t.ToTensor(),
#         ])
#         self.img_transform = t.Compose([
#             t.Lambda(lambda img: img.convert(color_space)),
#             t.Resize(img_size),
#             t.ToTensor(),
#             t.Normalize(self.stats_mean, self.stats_std)
#         ])  # Optional: Denoising, histogram equalization

#         assert split in ["train", "test"], "Only train and test split are available"
#         self.split = split
#         self.split_dir = os.path.join(self.dataset_dir, self.split)

#         self._load()
        

#     def __getitem__(self, index):
        
#         # Image preprocessing:
#         image = Image.open(self.image_paths[index])
#         image = self.img_transform(image)

#         # Mask preprocessing
#         mask = Image.open(self.mask_paths[index])
#         mask = self.mask_transform(mask)

#         return [image, mask]
#     def __len__(self):
#         return len(self.image_paths)
#     def _load(self):
#         image_dir = Path(os.path.join(self.split_dir, "image"))
#         mask_dir = Path(os.path.join(self.split_dir, "GT"))

#         self.image_paths = [img for img in image_dir.iterdir()]
#         self.mask_paths = [mask for mask in mask_dir.iterdir()]
        
     
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
