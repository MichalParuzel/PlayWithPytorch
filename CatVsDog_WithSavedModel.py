from torchvision import models, transforms
from CustomDataSet import CustomAnimal10DataSet
import torch.nn as nn
import torch
import os
import random
from PIL import Image, ImageFile


my_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ToCount:
    def __init__(self):
        self.total = 0
        self.wrong = 0
        self.write = 0

def make_prediction(model, path_to_image, label, to_count, mapping):
    model.eval()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(path_to_image).convert("RGB")
    #image.show()
    transformed_image = my_transform(image)
    transformed_image = transformed_image.unsqueeze(0) #this is adding the additional dimension
    output = model.forward(transformed_image)
    _, y_hat = output.max(1)

    #we_preditct = "Cat" if y_hat.item() == 0 else "Dog"
    we_preditct = mapping[y_hat.item()]

    to_count.total += 1
    if we_preditct == label:
        to_count.write += 1
    else:
        to_count.wrong += 1

def what_is_it(image_name: str):
    if image_name[0].isupper():
        return "Cat"
    return "Dog"

if __name__ == "__main__":
    path_to_model = r"C:\Users\HFD347\develp\pettest\Animal10_model.pth"
    #checkpoint = torch.load(path_to_model)
    model_d = torch.load(path_to_model)


    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(model_d)

    # Implement passing the image to the model
    #path_to_image = r"C:\Users\HFD347\develp\pettest\oxford-iiit-pet\images"
    path_to_image = r"C:\Users\HFD347\develp\Datasets\Animals-10"
    image_dataset = CustomAnimal10DataSet(path_to_image, my_transform)
    #list_of_images = os.listdir(path_to_image)
    #random.shuffle(list_of_images)

    mapping = {}
    for key in image_dataset.label_to_tensor_mapping:
        mapping[image_dataset.label_to_tensor_mapping[key].item()] = key

    list_of_images = image_dataset.all_images
    to_count = ToCount()
    idx = 0
    for image_name in list_of_images:
        #should_be = what_is_it(image_name)
        should_be = image_dataset.all_images_label_map[idx]

        image_full_path = os.path.join(path_to_image, should_be, image_name)
        #print("{0} is: ".format(image_name))
        make_prediction(model, image_full_path, should_be, to_count, mapping)
        idx += 1

    print("Total images: {0} of which {1} was wrong".format(to_count.total, to_count.wrong))

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



my_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
