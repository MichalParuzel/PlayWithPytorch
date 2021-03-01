from os import path, listdir
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = listdir(main_dir)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = path.join(self.main_dir, self.total_imgs[idx])
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        label = CustomDataSet.is_cat(self.total_imgs[idx])
        label = label.astype('float')
        return tensor_image, label

    @staticmethod
    def is_cat(image_name: str) -> np.ndarray:
        if image_name[0].isupper():
            return np.array(0)
        return np.array(1).astype(np.int64)

