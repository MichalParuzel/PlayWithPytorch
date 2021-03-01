from torchvision import models
from torchvision import transforms
import torch

from PIL import Image
from os import path

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

resnet = models.resnet34(pretrained=True)

image_path = r"C:\Users\HFD347\develp\pettest\oxford-iiit-pet\images"
cat_path = path.join(image_path, "Abyssinian_170.jpg")

img_cat = Image.open(cat_path).convert('RGB')

img_cat_preprocessed = preprocess(img_cat)
batch_img_cat_tensor = torch.unsqueeze(img_cat_preprocessed, 0)

resnet.eval()
out = resnet(batch_img_cat_tensor)

print(out)