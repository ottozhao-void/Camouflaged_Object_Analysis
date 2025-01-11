from Camouflage_Dataset import CamouflageDataset
from torch.utils.data import DataLoader
from utility import count_trainImg_stats
import torchvision.transforms as t
import argparse
import wandb


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size", type=int, default=(512, 512))

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
