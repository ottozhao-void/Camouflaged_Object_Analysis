import torch.utils.data as data
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as t

class CamouflageDataset(data.Dataset):
    def __init__(self, dataset_dir, split, transform = None):
        super().__init__()
        self.dataset_dir = dataset_dir

        self.transform = transform

        assert split in ["train", "split"], "Only train and test split are available"
        self.split = split
        self.split_dir = os.path.join(self.dataset_dir, self.split)

        self._load()
        

    def __getitem__(self, index):
        
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            image = t.ToTensor(Image.open(self.image_paths[index]))
            mask = t.ToTensor(Image.open(self.mask_paths[index]))

        return [image, mask]
    def __len__(self):
        return len(self.image_paths)
    def _load(self):
        image_dir = Path(os.path.join(self.split_dir, "image"))
        mask_dir = Path(os.path.join(self.split_dir, "GT"))

        self.image_paths = [img for img in image_dir.iterdir()]
        self.mask_paths = [mask for mask in mask_dir.iterdir()]