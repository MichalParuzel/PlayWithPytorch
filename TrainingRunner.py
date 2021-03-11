import HelperFunctions
from CustomDataSet import CustomDataSet, CustomAnimal10DataSet
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim

def data_preparation(pet_image_location_all, loss_function, batch_size=1, shuffle=True, num_workers=4, pretrained=True, fully_connected_layer_size=2):

    #this part will not work for CustomAnimal10DataSet -> TODO: figure out the way to make it generic
    #dataset_dict = CustomDataSet.split_to_train_validate_dataset(pet_image_location_all)

    #figure out how to make the dataset object type generic
    image_datasets = {x: CustomAnimal10DataSet(pet_image_location_all, dataset_dict[x], HelperFunctions.my_transform)
                  for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
               for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

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

    model_ft = HelperFunctions.train_model(model_ft, loss_function, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=1)

    '''
    path_to_model = r".\SavedModels\ModelWithCheckpoints.pth"
    model_ft = HelperFunctions.train_model_with_checkpoints(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device,
                                           dataset_sizes, path_to_model, num_epochs=10)
    #Save cat vs dog model:
    #path_to_model = r".\FirstModel.pth"
    #torch.save(model_ft.state_dict(), path_to_model)
    '''
