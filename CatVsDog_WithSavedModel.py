from torchvision import models, transforms

import torch.nn as nn
import torch
from PIL import Image, ImageFile

if __name__ == "__main__":
    path_to_model = r".\SavedModels\ModelWithCheckpoints.pth"
    checkpoint = torch.load(path_to_model)

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Implement passing the image to the model
    path_to_image = r"C:\Users\HFD347\develp\pettest\oxford-iiit-pet\images\Abyssinian_107.jpg"
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(path_to_image).convert("RGB")

    #Inference data transform operations:
    my_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transformed_image = my_transform(image)
    transformed_image = transformed_image.unsqueeze(0) #this is adding the additional dimension


    # Need to check how to convert this to probabilities of specific category
    output = model(transformed_image)
    print("Lets wait hear")

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



