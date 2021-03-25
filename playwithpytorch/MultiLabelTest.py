import pandas as pd
import torch
import os
from CustomDataSet import MoreGenericDataset
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

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

def get_pretrained_model(output_size, device):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, output_size)
    model_ft = model_ft.to(device)
    return model_ft

def train_multilabel(model, optimizer, loss_function, dataloader, epoches):
    model.train()

    for i in range(epoches):
        for input, label in dataloader:
            input.to(device)
            label.to(device)

            model.zero_grad() # check how to call it

            with torch.set_grad_enabled(True):
                output = model(input)
                loss = loss_function(output, label)
                loss.backward()
                optimizer.step()

    return model

def custom_loss_function(input, target):
    input = input.sigmoid()
    return -torch.where(target == 1, 1-input, input).log().mean()


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
    data_loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1)

    output_size = len(one_hot[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_pretrained_model(output_size, device)

    loss_func = custom_loss_function
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trained_model = train_multilabel(model, optimizer_ft, loss_func, data_loader, 1)


    print(t)