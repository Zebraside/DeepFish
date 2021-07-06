import segmentation_models_pytorch as smp

from lightning_modules import DeepFishLocalization, DeepFishDataModule

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def train(train_csv,
          test_csv,
          val_csv,
          image_dir,
          masks_dir):
    data_module = DeepFishDataModule(train_csv, test_csv, val_csv, image_dir, masks_dir, include_empty=False, batch_size=4)

    deepfish_model = DeepFishLocalization(model_head="PAN", backbone="efficientnet-b3")

    val_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/work/DeepFish/tmp',
        filename='PAN_b6_segmentation-{v_num}-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
    )

    early_stop = EarlyStopping(
        monitor='val_loss'
    )

    trainer = pl.Trainer(gpus=1,
                         precision=16,
                         max_epochs=100,
                         progress_bar_refresh_rate=5,
                         callbacks=[val_checkpoint_callback],
                         auto_scale_batch_size=False,
                         auto_lr_find=True)
    #trainer.tune(deepfish_model, datamodule=data_module)
    trainer.fit(deepfish_model, datamodule=data_module)


if __name__ == "__main__":
    BASE_DIR = "/work/data/DeepFish/Segmentation/"
    train(BASE_DIR + "train.csv",
          BASE_DIR + "test.csv",
          BASE_DIR + "val.csv",
          BASE_DIR + "images",
          BASE_DIR + "masks")
