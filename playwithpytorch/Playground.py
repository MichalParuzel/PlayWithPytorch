import  TrainingRunner
import torch.nn as nn

if __name__ == "__main__":
    path = r"C:\Users\HFD347\develp\Datasets\Animals-10"
    loss_function = nn.CrossEntropyLoss()
    TrainingRunner.data_preparation(path, loss_function, batch_size=1, shuffle=True, num_workers=4, pretrained=True, fully_connected_layer_size=10, num_epochs=1)