from os import path, listdir
from torch.utils.data import Dataset
import numpy as np
import random
from PIL import Image, ImageFile


class CustomDataSet(Dataset):
    def __init__(self, main_dir, img_list, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = img_list

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx) -> dict:
        image_name = self.total_imgs[idx]
        img_loc = path.join(self.main_dir, image_name)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        label = CustomDataSet.is_cat(image_name)
        label = label.astype('float')

        return tensor_image, label

    @staticmethod
    def is_cat(image_name: str) -> np.ndarray:
        if image_name[0].isupper():
            return np.array(0)
        return np.array(1).astype(np.int64)

    @staticmethod
    def split_to_train_validate_dataset(main_dir: str) -> dict:
        all_imgs = listdir(main_dir)
        list_len: int = len(all_imgs)
        validation_size: int = round(list_len*0.2)
        random.shuffle(all_imgs)
        validation_set: list = all_imgs[0:validation_size]
        train_set: list = all_imgs[validation_size:]
        output_mapping = {'val': [], 'train': []}

        for n in validation_set:
            output_mapping['val'].append(n)
        for n in train_set:
            output_mapping['train'].append(n)
        return output_mapping
