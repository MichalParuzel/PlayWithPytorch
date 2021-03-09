from os import path, listdir
from torch.utils.data import Dataset
import numpy as np
import random
import os
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

        return tensor_image, label

    @staticmethod
    def is_cat(image_name: str) -> np.ndarray:
        if image_name[0].isupper():
            return np.array(0).astype(np.int64) #check what this change will do
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


class CustomAnimal10DataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.all_images = []
        self.all_images_label_map = {}
        self.main_dir = main_dir
        self.transform = transform
        self.image_label_mapping = self.read_folders()
        self.label_mapping = self.map_labels()



    def __len__(self):
        return len(self.image_label_mapping)

    def __getitem__(self, idx):
        image_name = self.image_label_mapping[idx]
        img_loc = path.join(self.main_dir, image_name)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)

        label = np.array(self.label_mapping[idx]).astype(np.int64)
        return tensor_image, label

    def read_folders(self):
        dir_items = os.listdir(self.main_dir)
        labeled_dataset = {}
        for item in dir_items:
            if os.path.isdir(path.join(self.main_dir, item)):
                label = item
                category_dataset = os.listdir(path.join(self.main_dir, item))
                for cat_item in category_dataset:
                    if not labeled_dataset or label not in labeled_dataset:
                        labeled_dataset[label] = []
                    labeled_dataset[label].append(cat_item)
        return labeled_dataset


    #Need to test this one:
    def read_folders_and_map(self):
        dir_items = os.listdir(self.main_dir)
        idx = 0
        for item in dir_items:
            if os.path.isdir(path.join(self.main_dir, item)):
                label = item
                category_dataset = os.listdir(path.join(self.main_dir, item))
                for cat_item in category_dataset:
                    self.all_images.append(cat_item)
                    self.all_images_label_map[idx] = label

    def map_labels(self):
        self.label_mapping = {}
        key_list = [key for key in self.image_label_mapping]
        key_list.sort()
        idx = 0
        for key in key_list:
            self.label_mapping[key] = idx
            idx += 1

