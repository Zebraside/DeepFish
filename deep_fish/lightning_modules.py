import os

import torch
from torch.utils.data import DataLoader

import pandas as pd
import pytorch_lightning as pl

import albumentations as albu
import albumentations.pytorch as alto

from dataset import DeepFishDataset
from losses import DiceBCELoss, FocalTverskyLoss


def get_training_augmentation(img_size=640):
    train_transform = [
        albu.LongestMaxSize(img_size, always_apply=True),
        albu.PadIfNeeded(img_size, img_size, border_mode=0, always_apply=True),

        albu.HorizontalFlip(p=0.3),
        albu.VerticalFlip(p=0.2),
        albu.ToFloat(),
        alto.transforms.ToTensorV2()
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(img_size=640):
    test_transform = [
        albu.LongestMaxSize(img_size, always_apply=True),
        albu.PadIfNeeded(img_size, img_size, border_mode=0, always_apply=True),

        albu.ToFloat(),
        alto.transforms.ToTensorV2(),
    ]
    return albu.Compose(test_transform)


class DeepFishDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_csv,
                 test_csv,
                 val_csv,
                 image_dir,
                 masks_dir,
                 batch_size=2):
        super(DeepFishDataModule, self).__init__()
        self.batch_size = batch_size
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.val_csv = val_csv

        self.image_dir = image_dir
        self.masks_dir = masks_dir

    def prepare_data(self):
        train_imgs = pd.read_csv(self.train_csv)["ID"].tolist()
        test_imgs = pd.read_csv(self.test_csv)["ID"].tolist()
        val_imgs = pd.read_csv(self.val_csv)["ID"].tolist()

        train_img_paths = [os.path.join(self.image_dir, name + '.jpg') for name in train_imgs]
        train_mask_paths = [os.path.join(self.masks_dir, name + '.png') for name in train_imgs]

        test_img_paths = [os.path.join(self.image_dir, name + '.jpg') for name in test_imgs]
        test_mask_paths = [os.path.join(self.masks_dir, name + '.png') for name in test_imgs]

        val_img_paths = [os.path.join(self.image_dir, name + '.jpg') for name in val_imgs]
        val_mask_paths = [os.path.join(self.masks_dir, name + '.png') for name in val_imgs]

        self.train_set = DeepFishDataset(train_img_paths, train_mask_paths,
                                         augmentation=get_training_augmentation())

        self.val_set = DeepFishDataset(val_img_paths, val_mask_paths,
                                       augmentation=get_validation_augmentation())

        self.test_set = DeepFishDataset(test_img_paths, test_mask_paths,
                                       augmentation=get_validation_augmentation())

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=2, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2, drop_last=True)


class DeepFishSegmentator(pl.LightningModule):
    def __init__(self,
                 model):
        super(DeepFishSegmentator, self).__init__()
        self.model = model
        self.lr = 1e-3
        self.criterion = FocalTverskyLoss()

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        img, true_masks = batch
        logs = self.model(img)

        loss = self.criterion(logs, torch.unsqueeze(true_masks, 1))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, true_masks = batch
        logs = self.model(img)

        loss = self.criterion(logs, torch.unsqueeze(true_masks, 1))

        self.log('val_loss', loss, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        return (
            {'optimizer': optimizer, 'lr_scheduler': scheduler})
