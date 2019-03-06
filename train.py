import torch
import dataloader from utils
import * from visualizations

def train(classifier, segmenter, epochs=10):
    """Adversairially trains a segmentation model against a classifier
    
    Args:
        classifier (VGG16): a vgg-16 CNN
        segmentor (UNet): a unet segmentation model
        epochs (int): number of epochs to train
        
    Returns:
        classifier (VGG16): the trained classifier
        segmenter (UNet): the trained segmenter
    """
    
    data_loader = dataloader()
    loss = torch.nn.CrossEntropyLoss()
    gen_opt = torch.optim.Adam(segmenter.parameters())
    clas_opt = torch.optim.Adam(classifier.parameters())

    for i in range(epochs):
        for j, (imgs, lbls) in enumerate(data_loader):
            
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            
            #get the Grad-Cam and Guided BP for the batch of images
            gc = grad_cam(classifier, imgs)
            gbp = guided_bp(classifier, imgs)
            
            seg_input = torch.cat((imgs, gc, gbp), 1).cuda()
            
            #Get output masks for correct class
            seg_output = segmenter(seg_input)[:,lbls,:,:]
            
            #Flip the mask so that background is segmented
            masks = (seg_output < 0.9).type(torch.cuda.FloatTensor)
            
            seg_imgs = []
            
            #Extract segmented background
            for k in range(imgs.shape[0]):
                seg_imgs.append(imgs[k] * masks[k,k])
            seg_imgs = torch.stack(seg_imgs)
            
            #segmenter wants to maximize classifier loss after removing the target object
            s_loss = -loss(classifier(seg_imgs), lbls)
            
            gen_opt.zero_grad()
            s_loss.backward()
            gen_opt.step()
            
            #Classifier wants to minimize loss on original images and segmented images
            c_loss = loss(classifier(imgs), lbls) + loss(classifier(seg_imgs), lbls)
            
            clas_opt.zero_grad()
            c_loss.backward()
            clas_opt.step()
            
            if j % 10 == 0:
                print("Epoch {} Iter {} - Segmentation Loss: {}     Classificaton Loss: {}".format(i, j, s_loss*-1, c_loss))
            
    return classifier, segmenter