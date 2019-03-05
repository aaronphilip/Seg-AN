import torch
from torch.autograd import Variable
import numpy as np
import cv2

def grad_cam(model, img_arr):
    """Generates a batch of grad-CAMs
    
    Args:
        model (VGG16): A cuda enabled VGG16 model
        img_arr (Tensor): a normalized batch of images
        
    Returns: 
        grad_cams (Tensor): A batch of grad-CAMs
    
    """
    grad_preds = model(img_arr)
    
    #Obtain the logit value of the predicted class for each image
    idx = torch.from_numpy(np.argmax(grad_preds.cpu().data.numpy(), axis=-1)).cuda()
    grad_preds = torch.stack([a[i]  for a, i in zip(grad_preds, idx)])
    
    grad_cams = []
    
    for i, grad_pred in enumerate(grad_preds):
        
        #backprop for one image classification
        model.classifier.zero_grad()
        model.features.zero_grad()
        grad_pred.backward(retain_graph=True)
        
        #Obtain the output of the last convolutional layer
        conv_output = model.last_conv.cpu().data.numpy()[i]
        
        #Obtain the gradients for the last convolutional layer
        gradients = model.gradient[-1].cpu().data.numpy()[i]
        
        #pool gradients across channels
        weights = np.mean(gradients, axis = (1,2))
        
        grad_cam = np.zeros(conv_output.shape[ 1 :], dtype=np.float32)
        
        #Weigh each channel in conv_ouput
        for i, weight in enumerate(weights):
            grad_cam += weight * conv_output[i, :, :]
        
        #normalize the grad-CAM
        grad_cam = np.maximum(grad_cam, 0)
        grad_cam = cv2.resize(grad_cam, (224,224))
        grad_cam = grad_cam - np.min(grad_cam)
        grad_cam = grad_cam / np.max(grad_cam)
        grad_cam = torch.Tensor(grad_cam)
        
        grad_cams.append(grad_cam)
    
    grad_cams = torch.stack(grad_cams).unsqueeze(1).cuda()
    
    return grad_cams

def guided_bp(model, img_arr):
    """Generates a batch of guided-backprops
    
    Args:
        model (VGG16): A cuda enabled VGG16 model
        img_arr (Tensor): a normalized batch of images
        
    Returns: 
        guided_bps (Tensor): A batch of guided_backprops    
    
    """
    img_arr = Variable(img_arr, requires_grad=True)
    
    gbp_preds = model(img_arr, guided=True)

    #Obtain the logit value of the predicted class for each image
    idx = torch.from_numpy(np.argmax(gbp_preds.cpu().data.numpy(), axis=-1)).cuda()
    gbp_preds = torch.stack([a[i]  for a, i in zip(gbp_preds, idx)])
    
    guided_bps = []
    
    for i, gbp_pred in enumerate(gbp_preds):
        #backprop for one image classification
        model.classifier.zero_grad()
        model.features.zero_grad()
        gbp_pred.backward(retain_graph=True)
        
        #obtain the gradient w.r.t. to the image
        guided_bp = img_arr.grad[i]
        guided_bps.append(guided_bp)
    
    guided_bps = torch.stack(guided_bps).cuda()
    
    return guided_bps