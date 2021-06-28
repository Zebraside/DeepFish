from torch.utils.data import Dataset, DataLoader
import os
import cv2


class DeepFishDataset(Dataset):
    def __init__(self, img_paths, label_paths, augmentation=None):
        # img_paths and label_paths must be synchronized
        self.input_images = img_paths
        self.target_masks = label_paths

        self.aug = augmentation

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = cv2.imread(self.input_images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.target_masks[idx], cv2.IMREAD_GRAYSCALE)

        mask = mask / 255

        if self.aug:
            sample = self.aug(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
