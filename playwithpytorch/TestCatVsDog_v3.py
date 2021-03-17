import HelperFunctions
from CustomDataSet import CustomDataSet, CustomAnimal10DataSet
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim

if __name__ == "__main__":
    '''
        This is the place where we should put the definition of where the data is
        Write custom dataloader
        Get the dataset size -> there is a dependency for that in train method
        class_names -> wtf? 
        Than there is a torch.device, which is well, device
    '''

    pet_image_location_all = r"C:\Users\HFD347\develp\Datasets\Animals-10"
    dataset = CustomAnimal10DataSet(pet_image_location_all, HelperFunctions.my_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16)
    dataset_size = len(dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = HelperFunctions.load_model(device, 10, True, path_to_model=r"C:\Users\HFD347\develp\PlayWithPytorch\playwithpytorch\TrainedModels\changed_training.pth")
    #model_ft = HelperFunctions.load_model(device, 10, False)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #HelperFunctions.validate_model(model_ft, criterion, dataloader, device, dataset_size, 100)
    trained_model = HelperFunctions.train_model_new(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloader, device, dataset_size, num_epochs=1, print_every=1600)
    #model_ft = HelperFunctions.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloader, device, dataset_size, num_epochs=1)

    '''
    path_to_model = r".\SavedModels\ModelWithCheckpoints.pth"
    model_ft = HelperFunctions.train_model_with_checkpoints(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device,
                                           dataset_sizes, path_to_model, num_epochs=10)
    #Save cat vs dog model:
    #path_to_model = r".\FirstModel.pth"
    #torch.save(model_ft.state_dict(), path_to_model)
    '''


    print("Haha it is done")