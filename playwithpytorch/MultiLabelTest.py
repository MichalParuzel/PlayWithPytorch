import pandas as pd
import torch
import os
from CustomDataSet import MoreGenericDataset
from torchvision import datasets, models, transforms

def explore_dataset():
    path_to_data = r"C:\Users\HFD347\.fastai\data\pascal_2007"
    df = pd.read_csv("{0}\\test.csv".format(path_to_data))
    #print(df.head())
    #print(df['labels'])
    #making distinct list of categories
    cat_list = [labels for labels in df['labels']]
    label_set = set()
    for labels in cat_list:
        for label in labels.split(' '):
            label_set.add(label)

    label_list = list(label_set)
    label_list.sort()
    print(label_list)

def generate_encoder_mapping(path_to_data, label_column_name):
    df = pd.read_csv("{0}\\test.csv".format(path_to_data))
    cat_list = [labels for labels in df[label_column_name]]
    label_set = set()
    for labels in cat_list:
        for label in labels.split(' '):
            label_set.add(label)

    label_list = list(label_set)
    label_list.sort()
    return {label_list[idx]: idx for idx in range(len(label_list))}

def generate_one_hot_encoding(path_to_data, label_column_name):
    df = pd.read_csv("{0}\\test.csv".format(path_to_data))
    cat_list = [labels for labels in df[label_column_name]]
    label_set = set()
    for labels in cat_list:
        for label in labels.split(' '):
            label_set.add(label)

    label_list = list(label_set)
    label_list.sort()

    label_mapping = {label_list[idx]: idx for idx in range(len(label_list))}
    one_hot_len = len(label_list)
    one_hot_encodings = []
    for labels in cat_list:
        one_hot = torch.zeros(one_hot_len)
        one_hot_encodings.append(one_hot)
        for label in labels.split(' '):
            idx = label_mapping[label]
            one_hot[idx] = 1
    return one_hot_encodings

def generate_dataset(path_to_data, img_list, label_list, img_subfolders, transform):
    return MoreGenericDataset(path_to_data, img_list, label_list, img_subfolders, transform)


def get_image_list(path_to_images):
    dir_items = os.listdir(path_to_images)
    for item in dir_items:
        yield item

test_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])  # adjust this later to proper values for dataset

if __name__ == "__main__":
    path_to_data = r"C:\Users\HFD347\.fastai\data\pascal_2007"
    labels_path = os.path.join(path_to_data, "test.csv")
    img_list = list(get_image_list(os.path.join(path_to_data, "test")))

    one_hot = generate_one_hot_encoding(path_to_data, "labels")
    ds = generate_dataset(path_to_data, img_list, one_hot, "test", test_transform)
    print(t)