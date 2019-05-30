from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torchvision.datasets as datasets
import torch

def preprocess_img(img_path):
    """Normalizes an image for passing through a pretrained network
    Args:
        img_path (str): path to the image file
        
    Returns:
        img (numpy array): an image converted to a numpy array
        img_arr (Tensor): normalized image Tensor
    """
    img = Image.open(img_path).resize((224,224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    img_arr = Variable(normalize(transforms.ToTensor()(img)).unsqueeze(0), requires_grad=True)
    img = np.asarray(img)
    
    return img, img_arr

def dataloader():
    """Data loader for loading images and labels
    
    https://github.com/pytorch/examples/tree/master/imagenet
    
    Returns:
        data_loader (DataLoader): a PyTorch DataLoader to load in imagenet
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.VOCSegmentation(
        './voc', 
        year='2012', 
        image_set='trainval', 
        download=True,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]))

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    return data_loader