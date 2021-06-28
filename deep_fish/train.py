import segmentation_models_pytorch as smp

from lightning_modules import DeepFishSegmentator, DeepFishDataModule

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def train(train_csv,
          test_csv,
          val_csv,
          image_dir,
          masks_dir):
    data_module = DeepFishDataModule(train_csv, test_csv, val_csv, image_dir, masks_dir)

    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )

    deepfish_model = DeepFishSegmentator(model)

    val_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/content/tmp/check/',
        filename='PAN_EFB5_452_best_val-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
    )

    early_stop = EarlyStopping(
        monitor='val_loss'
    )

    trainer = pl.Trainer(gpus=0,
                         precision=32,
                         max_epochs=50,
                         progress_bar_refresh_rate=5,
                         callbacks=[val_checkpoint_callback, early_stop],
                         auto_scale_batch_size='binsearch')

    trainer.fit(deepfish_model, data_module)


if __name__ == "__main__":
    BASE_DIR = "C:\\Users\\Kirill_Molchanov\\Downloads\\DeepFish\\Localization\\"
    train(BASE_DIR + "train.csv",
          BASE_DIR + "test.csv",
          BASE_DIR + "val.csv",
          BASE_DIR + "images",
          BASE_DIR + "masks")
