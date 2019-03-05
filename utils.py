from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torchvision.datasets as datasets
import torch

def preprocess_img(img_path):
    img = Image.open(img_path).resize((224,224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    img_arr = Variable(normalize(transforms.ToTensor()(img)).unsqueeze(0), requires_grad=True)
    img = np.asarray(img)
    
    return img, img_arr

def dataloader():
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    traindir = "tiny-imagenet-200/train"

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))
    

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, pin_memory=True)
    
    return data_loader