#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-30

import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data


class _BaseDataset(data.Dataset):
    """
    Base dataset class
    """

    def __init__(
        self,
        root,
        split,
        mean_bgr=None,
        augment=True,
        base_size=None,
        crop_size=321,
        scales=(1.0),
        flip=True,
        task = None
    ):
        self.root = root
        self.split = split
        self.mean_bgr = np.array(mean_bgr) if isinstance(mean_bgr, tuple) else mean_bgr
        self.augment = augment
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip
        self.files = []
        self._set_files()
        self.task = task

        cv2.setNumThreads(0)

    def _set_files(self):
        """
        Create a file path/image id list.
        """
        raise NotImplementedError()

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        raise NotImplementedError()

    def _augmentation(self, image, label):
        
        image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label).resize((self.crop_size, self.crop_size), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)

        # Random flipping
        if random.random() < 0.5:  # Horizontal flip
            image = np.fliplr(image).copy()  # HWC
            label = np.fliplr(label).copy()  # HW

        if random.random() < 0.5:  # Vertical flip
            image = np.flipud(image).copy()  # HWC
            label = np.flipud(label).copy()  # HW

        return image, label

    def __getitem__(self, index):
        image, label = self._load_data(index)
        if self.augment:
            image, label = self._augmentation(image, label)
        # HWC -> CHW
        image = image.transpose(2, 0, 1)

        return image.astype(np.float32) / 255.0, label.astype(np.int64) / 255

    def __len__(self):
        return len(self.image_list)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
