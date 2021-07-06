import os

import torch
from torch.utils.data import DataLoader

import pandas as pd
import pytorch_lightning as pl

import albumentations as albu
import albumentations.pytorch as alto

import segmentation_models_pytorch as smp

from dataset import DeepFishDataset
from losses import DiceBCELoss, FocalTverskyLoss, FocalLoss


def get_training_augmentation(img_size=960, keep_aspect_ratio=True):
    train_transform = [
        albu.LongestMaxSize(img_size, always_apply=True),
        # albu.PadIfNeeded(img_size, img_size, border_mode=0, always_apply=True),
        # albu.Resize(img_size, img_size), 

        albu.HorizontalFlip(p=0.3),
        albu.VerticalFlip(p=0.2),
        albu.ToFloat(),
        alto.transforms.ToTensorV2()
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(img_size=960, keep_aspect_ratio=True):
    test_transform = [
        albu.LongestMaxSize(img_size, always_apply=True),
        #albu.PadIfNeeded(img_size, img_size, border_mode=0, always_apply=True),
        # albu.Resize(img_size, img_size),

        albu.ToFloat(),
        alto.transforms.ToTensorV2()
    ]
    return albu.Compose(test_transform)


class DeepFishDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_csv,
                 test_csv,
                 val_csv,
                 image_dir,
                 masks_dir,
                 include_empty=True,
                 batch_size=2):
        super(DeepFishDataModule, self).__init__()
        self.batch_size = batch_size
        self.include_empty = include_empty
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.val_csv = val_csv

        self.image_dir = image_dir
        self.masks_dir = masks_dir

    def prepare_data(self):
        train_imgs = pd.read_csv(self.train_csv)["ID"].tolist()
        test_imgs = pd.read_csv(self.test_csv)["ID"].tolist()
        val_imgs = pd.read_csv(self.val_csv)["ID"].tolist()

        print("TRAIN IMGS LEN", len(train_imgs))
        print("TEST IMGS LEN", len(test_imgs))
        print("VAL IMGS LEN", len(val_imgs))

        train_img_paths = [os.path.join(self.image_dir, name + '.jpg') for name in train_imgs if self.include_empty or 'empty' not in name]
        train_mask_paths = [os.path.join(self.masks_dir, name + '.png') for name in train_imgs if self.include_empty or 'empty' not in name]

        test_img_paths = [os.path.join(self.image_dir, name + '.jpg') for name in test_imgs if self.include_empty or 'empty' not in name]
        test_mask_paths = [os.path.join(self.masks_dir, name + '.png') for name in test_imgs if self.include_empty or 'empty' not in name]

        val_img_paths = [os.path.join(self.image_dir, name + '.jpg') for name in val_imgs if self.include_empty or 'empty' not in name]
        val_mask_paths = [os.path.join(self.masks_dir, name + '.png') for name in val_imgs if self.include_empty or 'empty' not in name]

        self.train_set = DeepFishDataset(train_img_paths, train_mask_paths,
                                         augmentation=get_training_augmentation())

        self.val_set = DeepFishDataset(val_img_paths, val_mask_paths,
                                       augmentation=get_validation_augmentation())

        self.test_set = DeepFishDataset(test_img_paths, test_mask_paths,
                                       augmentation=get_validation_augmentation())

        print("DS LEN TR, VAL, TEST", len(self.train_set), len(self.val_set), len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=True)


class DeepFishLocalization(pl.LightningModule):
    def __init__(self, model_head="PAN", backbone="efficientnet-b5", loss="dice_bce", activation=None):
        super(DeepFishLocalization, self).__init__()

        if model_head == "PAN":
            self.model = smp.PAN(
                encoder_name=backbone,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation=activation
                )
        elif model_head == "UNet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation=activation
                )
        elif model_head == "UnetPlusPlus":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation=activation
                )
        else:
            raise RuntimeError("Unknown model head")
            
        self.lr = 0.001

        if loss == "focal":
            self.criterion = FocalLoss()
        elif loss == "tversky":
            self.criterion = FocalTverskyLoss()
        elif loss == "dice_bce":
            self.criterion = DiceBCELoss()

        self.save_hyperparameters()

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        img, true_masks = batch
        logs = self.model(img)
        logs = torch.squeeze(logs.permute(0, 2, 3, 1))
        logs = torch.squeeze(logs)

        loss = self.criterion(logs, true_masks)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, true_masks = batch
        logs = self.model(img)
        logs = torch.squeeze(logs.permute(0, 2, 3, 1))
        logs = torch.squeeze(logs)

        loss = self.criterion(logs, true_masks)


        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        return (
            {'optimizer': optimizer}) #, 'lr_scheduler': scheduler, 'monitor': 'train_loss'})
