import pandas as pd
import torch
import os
from CustomDataSet import MoreGenericDataset
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import math
import copy
import time

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

def generate_one_hot_encoding(path_to_data, file_name, label_column_name):
    df = pd.read_csv("{0}\\{1}".format(path_to_data, file_name))
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

def generate_mapping(path_to_data, file_name, label_column_name):
    df = pd.read_csv("{0}\\{1}".format(path_to_data, file_name))
    cat_list = [labels for labels in df[label_column_name]]
    label_set = set()
    for labels in cat_list:
        for label in labels.split(' '):
            label_set.add(label)

    label_list = list(label_set)
    label_list.sort()

    return {idx: label_list[idx] for idx in range(len(label_list))}


def generate_dataset(path_to_data, img_list, label_list, img_subfolders, transform):
    return MoreGenericDataset(path_to_data, img_list, label_list, img_subfolders, transform)


def get_image_list(path_to_images):
    dir_items = os.listdir(path_to_images)
    for item in dir_items:
        yield item

def get_pretrained_model(output_size, device, pre_trained=True):
    model_ft = models.resnet18(pretrained=pre_trained)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, output_size)
    model_ft = model_ft.to(device)
    return model_ft

def train_multilabel(model, optimizer, loss_function, dataloader, epoches, path_to_save):
    model.train()
    best_model_wts = copy.deepcopy(model.state_dict())
    since = time.time()

    for i in range(epoches):
        acc_sum = 0.0
        pred_count = 0
        accur = 0.0
        best_acc = 0.0
        for input, label in dataloader:
            input.to(device)
            label.to(device)

            model.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(input)
                loss = loss_function(output, label)
                loss.backward()
                optimizer.step()

            acc_sum += calc_accuracy(output, label)
            pred_count += 1
            if pred_count % 1024 == 0:
                print("After {} preds accuracy is: {:.4f}%".format(pred_count, acc_sum/pred_count*100))
            #print("Current accuracy: {0}".format(curr_acc))
        accur = (acc_sum / pred_count)*100
        print("Accuracy: {:.4f}%".format(accur))
        if accur > best_acc:
            best_acc = accur
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Saving model")
            curr_model_name = "Multilabel_after_" + str(i) + "_epochs.pth"
            path_to_save = ""
            torch.save(model.state_dict(), os.path.join(path_to_save, curr_model_name))
        model.load_state_dict(best_model_wts)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model

def validate_multilabel(model, loss_function, dataloader, device):
    model.eval()
    since = time.time()

    acc_sum = 0.0
    pred_count = 0
    accur = 0.0
    best_acc = 0.0

    for input, label in dataloader:
        input.to(device)
        label.to(device)

        model.zero_grad()

        output = model(input)
        loss = loss_function(output, label)
        #print("Validation Loss is: {}, Accuracy is: {}".format(loss, calc_accuracy(output, label)))

        acc_sum += calc_accuracy(output, label)
        pred_count += 1
        if pred_count % 100 == 0:
            print("After {} preds accuracy is: {:.4f}%".format(pred_count, acc_sum/pred_count*100))
        #print("Current accuracy: {0}".format(curr_acc))

    accur = (acc_sum / pred_count)*100
    print("Final Accuracy: {:.4f}%".format(accur))

    time_elapsed = time.time() - since
    print('Validation completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def calc_accuracy(pred, target, threshhold=0.5):
    pred_norm = pred.sigmoid()
    return ((pred_norm > threshhold) == target.bool()).float().mean()

# TODO: finish this function to get the predicted vs target labels
def predicted_should_be(pred, target, label_mapping, threshhold=0.5):
    pred_norm = pred.sigmoid()
    predicted = (pred_norm > threshhold).bool().float()


def custom_loss_function(input, target):
    input = input.sigmoid()
    return -torch.where(target == 1, 1-input, input).log().mean()

def custom_loss_function_2(input, target):
    input = input.sigmoid()

    tmp = 0
    for i in range(len(input.T)):
        tmp += input.T[i].item() * math.log(target.T[i].item())

    return -1 * tmp

test_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])  # adjust this later to proper values for dataset


def load_training():
    path_to_save_model = r".\TrainedModels"
    path_to_data = r"C:\Users\HFD347\.fastai\data\pascal_2007"
    labels_path = os.path.join(path_to_data, "train.csv")
    img_list = list(get_image_list(os.path.join(path_to_data, "train")))

    one_hot = generate_one_hot_encoding(path_to_data, "train.csv", "labels")
    ds = generate_dataset(path_to_data, img_list, one_hot, "train", test_transform)
    data_loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=1)

    output_size = len(one_hot[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_pretrained_model(output_size, device)

    loss_func = custom_loss_function
    pytorch_cross_entropy = nn.BCEWithLogitsLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trained_model = train_multilabel(model, optimizer_ft, pytorch_cross_entropy, data_loader, 3, path_to_save_model)

def load_validation():
    path_to_trained_model = r".\TrainedModels\WithProperLossFunc\CurrentBest.pth"
    path_to_data = r"C:\Users\HFD347\.fastai\data\pascal_2007"
    labels_path = os.path.join(path_to_data, "test.csv")
    img_list = list(get_image_list(os.path.join(path_to_data, "test")))
    one_hot = generate_one_hot_encoding(path_to_data, "test.csv", "labels")
    label_mapping = generate_mapping(path_to_data, "test.csv", "labels")
    ds = generate_dataset(path_to_data, img_list, one_hot, "test", test_transform)
    data_loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1)

    output_size = len(one_hot[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_pretrained_model(output_size, device, False)
    model_state = torch.load(path_to_trained_model, map_location=device)
    model.load_state_dict(model_state)

    pytorch_cross_entropy = nn.BCEWithLogitsLoss()
    validate_multilabel(model, pytorch_cross_entropy, data_loader, device)

if __name__ == "__main__":
    load_validation()

# TODO: Create a method to gather statistics on which labels were misspredicted and the threshold margin