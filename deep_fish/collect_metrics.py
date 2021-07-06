import math

import tqdm
import torch
import numpy as np
import cv2

import segmentation_models_pytorch as smp
from pytorch_toolbelt.losses import DiceLoss

from utils.convert_location_masks import count_objects
from lightning_modules import DeepFishLocalization, DeepFishDataModule


def count_from_mask(mask):
    return count_objects(mask)

def localization_RMSE(model, dataloader, count_func=count_from_mask):
    result = []
    gt = []
    act = []
    counter = 0
    abs_error = []
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        img, mask = batch
        assert(len(img) == 1)

        output = model(img)

        #print("OUTPUT", np.min((output * 255)[0].numpy().astype(np.uint8)), np.max((output * 255)[0].numpy().astype(np.uint8)))

        gt_count = count_from_mask((mask * 255)[0].numpy().astype(np.uint8))
        actual_count = count_from_mask((output * 255)[0][0].numpy().astype(np.uint8))
        # print(output.shape, mask.shape)
        # cv2.imwrite(f"/work/DeepFish/tmp/result_{counter}.jpg", (output * 255)[0][0].numpy().astype(np.uint8)) # (output * 255)[0].numpy().astype(np.uint8)
        # counter += 1
        # print("ACT COUNT", actual_count)
        gt.append(gt_count)
        act.append(actual_count)
        result.append(math.pow(gt_count - actual_count, 2))
        abs_error.append(abs(gt_count - actual_count))

    RMSE = math.sqrt(sum(result) / len(result))
    abs_error = sum(abs_error) / len(abs_error)
    gt = np.array(gt)
    act = np.array(act)
    
    print("Localization RMSE: ", RMSE)
    print("Count abs error: ", abs_error)
    print(f'GT avg: {np.sum(gt) / len(gt)} max: {np.max(gt)} min: {np.min(gt)}')
    print(f'Actual avg: {np.sum(act) / len(act)} max: {np.max(act)} min: {np.min(act)}')
    print("Avg err:", np.sum(np.abs(gt - act)) / len(gt) )


def collect_count_metrics(data_csv,
                          images_dir,
                          masks_dir,
                          checkpoint_path,
                          include_empty=False):
    with torch.no_grad():                          
        data_module = DeepFishDataModule(data_csv, data_csv, data_csv, images_dir, masks_dir, include_empty=include_empty, batch_size=1)
        data_module.prepare_data()

        model = smp.PAN(
            encoder_name="efficientnet-b5",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )


        deepfish_model = DeepFishLocalization.load_from_checkpoint(checkpoint_path)
        deepfish_model.eval()
        localization_RMSE(deepfish_model, data_module.test_dataloader())

def collect_segmentation(data_csv,
                          images_dir,
                          masks_dir,
                          checkpoint_path,
                          include_empty=False):

    with torch.no_grad():  
        data_module = DeepFishDataModule(data_csv, data_csv, data_csv, images_dir, masks_dir, include_empty=include_empty, batch_size=1)
        data_module.prepare_data()
        dataloader = data_module.test_dataloader()

        model = DeepFishLocalization.load_from_checkpoint(checkpoint_path)
        model.eval()

        dice_loss_fcn = DiceLoss(mode="binary", from_logits=True)
        dice_coef = []
        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            img, mask = batch
            assert(len(img) == 1)

            output = model(img)
            #output = torch.squeeze(output.permute(0, 2, 3, 1))
            #print(output.shape, mask.shape)

            dice_loss = dice_loss_fcn(output, torch.unsqueeze(mask, 0))
            dice_coef.append(1 - dice_loss)

        print("Avg Dice coef:", sum(dice_coef) / len(dice_coef))
        
 

if __name__ == "__main__":
    # BASE_DIR = "/work/data/DeepFish/Localization/"
    # collect_count_metrics(BASE_DIR + "test.csv",
    #                       BASE_DIR + "images",
    #                       BASE_DIR + "9_masks",
    #                       "/work/DeepFish/tmp/PAN_b6_segmentation-epoch=82-val_loss=0.2182.ckpt")

    # BASE_DIR = "/work/data/DeepFish/Segmentation/"
    # collect_segmentation(BASE_DIR + "test.csv",
    #                       BASE_DIR + "images",
    #                       BASE_DIR + "masks",
    #                       "/work/DeepFish/tmp/PAN_b6_segmentation-epoch=33-val_loss=0.2577.ckpt")