from Camouflage_Dataset import CamouflageDataset
from torch.utils.data import DataLoader
from utility import count_trainImg_stats
import torchvision.transforms as t


if __name__ == "__main__":
    transform = t.Compose([
        t.Resize((512, 512)),
        t.ToTensor()
    ])
    # mean = [0.0228, 0.0218, 0.0172], std = [0.0098, 0.0095, 0.0089]
    count_trainImg_stats(
        dataset_dir="/data1/zhaofanghan/Advanced_ML_Coursework/dataset",
        transform=transform
    )