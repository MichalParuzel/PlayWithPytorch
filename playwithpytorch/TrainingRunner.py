#import HelperFunctions
#from CustomDataSet import CustomDataSet, CustomAnimal10DataSet
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim


def data_preparation(image_dataset, pet_image_location_all, loss_function, batch_size=1, shuffle=True, num_workers=4, pretrained=True, fully_connected_layer_size=2, num_epochs=1):
    dataloader = {"train": DataLoader(image_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)}
    dataset_size = {"train": len(image_dataset)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # add mechanism to make the model generic
    model_ft = models.resnet18(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    HelperFunctions.freeze_all_layers(model_ft)
    model_ft.fc = nn.Linear(num_ftrs, fully_connected_layer_size)
    model_ft = model_ft.to(device)

    # Figure out to make those generic as well
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = HelperFunctions.train_model(model_ft, loss_function, optimizer_ft, exp_lr_scheduler, dataloader, device, dataset_size, num_epochs)

    path_to_model = r".\Animal10_model.pth"
    torch.save(model_ft.state_dict(), path_to_model)

