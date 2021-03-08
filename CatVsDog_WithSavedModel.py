from torchvision import models, transforms

import torch.nn as nn
import torch
import os
from PIL import Image, ImageFile

my_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def make_prediction(model, path_to_image):
    model.eval()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(path_to_image).convert("RGB")
    transformed_image = my_transform(image)
    transformed_image = transformed_image.unsqueeze(0) #this is adding the additional dimension
    output = model.forward(transformed_image)
    _, y_hat = output.max(1)

    if y_hat.item() == 0:
        print("We got a cat!")
    elif y_hat.item() == 1:
        print("We got a dog!")
    else:
        print("What do we have???")


if __name__ == "__main__":
    path_to_model = r".\SavedModels\ModelWithCheckpoints.pth"
    checkpoint = torch.load(path_to_model)

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Implement passing the image to the model
    path_to_image = r"C:\Users\HFD347\develp\pettest\oxford-iiit-pet\images"
    list_of_images = os.listdir(path_to_image)

    for image_name in list_of_images:
        image_full_path = os.path.join(path_to_image, image_name)
        print("{0} is: ".format(image_name))
        make_prediction(model, image_full_path)

'''
    # For now lets leave it and see how the checkpoint looks like
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    HelperFunctions.freeze_all_layers(model)

    model.fc = nn.Linear(num_ftrs, 2)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    model.eval()
'''



