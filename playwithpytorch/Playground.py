import TrainingRunner
import torch.nn as nn
import torch
import HelperFunctions

if __name__ == "__main__":
    #path = r"C:\Users\HFD347\develp\Datasets\Animals-10"
    #loss_function = nn.CrossEntropyLoss()
    #image_dataset = CustomAnimal10DataSet(pet_image_location_all, HelperFunctions.my_transform)
    #TrainingRunner.data_preparation(image_dataset, path, loss_function, batch_size=1, shuffle=True, num_workers=4, pretrained=True, fully_connected_layer_size=10, num_epochs=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_to = r"C:\Users\HFD347\develp\PlayWithPytorch\playwithpytorch\TrainedModels\changed_training.pth"
    model_ft = HelperFunctions.load_model(device, 10, True, path_to)

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list

    #print(model_ft)
    counter = 0
    model_children = list(model_ft.children())
    for idx in range(len(model_children)):
        if type(model_children[idx]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[idx].weight)
            conv_layers.append(model_children[idx])
        elif type(model_children[idx]) == nn.Sequential:
            for j in range(len(model_children[idx])):
                for child in model_children[idx][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    print("Total convolutional layers: {}".format(counter))
