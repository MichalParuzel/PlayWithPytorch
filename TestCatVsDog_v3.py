import HelperFunctions
from CustomDataSet import CustomDataSet
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
    pet_image_location_all = r"C:\Users\HFD347\develp\pettest\oxford-iiit-pet\images"
    dataset_dict = CustomDataSet.split_to_train_validate_dataset(pet_image_location_all)

    image_datasets = {x: CustomDataSet(pet_image_location_all, dataset_dict[x], HelperFunctions.my_transform)
                      for x in ['train', 'val']}

    #train_loader = DataLoader(dataset=image_datasets,
    #                                     batch_size=4,
    #                                     shuffle=True,
    #                                     num_workers=1)
    #dict_key = "train"
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    #class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''
        So next part is coming hear, loading the pretrained model
    '''

    # Definitely need to check this part, it looks it is freezing the layer, so need to make sure how it works
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    HelperFunctions.freeze_all_layers(model_ft)

    model_ft.fc = nn.Linear(num_ftrs, 2)


    #
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = HelperFunctions.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=1)

    '''
    path_to_model = r".\SavedModels\ModelWithCheckpoints.pth"
    model_ft = HelperFunctions.train_model_with_checkpoints(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device,
                                           dataset_sizes, path_to_model, num_epochs=10)
    #Save cat vs dog model:
    #path_to_model = r".\FirstModel.pth"
    #torch.save(model_ft.state_dict(), path_to_model)
    '''


    print("Haha it is done")