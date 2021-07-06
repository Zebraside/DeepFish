import os

from torchvision.utils import save_image

from lightning_modules import DeepFishDataModule


def save_for_loader(loader, save_dir, prefix):
    imgs, masks = next(iter(loader))
    print("IMG", imgs.shape, "MASKS", (masks * 255).shape)
    masks = masks.unsqueeze(1)
    save_image(imgs, os.path.join(save_dir, f"{prefix}_imgs.png"))
    save_image((masks * 255), os.path.join(save_dir, f"{prefix}_masks.png"))
    save_image(imgs + masks * 255, f"{prefix}_with_masks.png")

def visualize(train_csv,
              test_csv,
              val_csv,
              image_dir,
              masks_dir,
              save_dir):
    data_module = DeepFishDataModule(train_csv, test_csv, val_csv, image_dir, masks_dir, include_empty=False, batch_size=2)
    data_module.prepare_data()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    save_for_loader(train_loader, save_dir, "train")
    save_for_loader(val_loader, save_dir, "val")
    save_for_loader(test_loader, save_dir, "test")


if __name__ == "__main__":
    BASE_DIR = "/work/data/DeepFish/Localization/"
    visualize(BASE_DIR + "train.csv",
          BASE_DIR + "val.csv",
          BASE_DIR + "test.csv",
          BASE_DIR + "images",
          BASE_DIR + "9_masks",
          "/work/DeepFish/tmp")