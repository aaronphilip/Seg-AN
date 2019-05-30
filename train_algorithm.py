import torch
from utils import dataloader 
from visualizations import *

class_dict = {1:"airplane",       2:"bicycle",          3:"bird", 
              4:"boat",           5:"bottle",           6:"bus",
              7:"car",            8:"cat",              9:"chair",
             10:"cow",           11: "dining table",   12: "dog",
             13: "horse",        14:"motorbike",       15:"person",
             16: "potted plant", 17:"sheep",           18:"sofa",
             19:"train",         20:"tv/monitor"}

def train(classifier, segmenter, epochs=10):
    data_loader = dataloader()
    loss = torch.nn.CrossEntropyLoss()
    gen_opt = torch.optim.Adam(segmenter.parameters())
    clas_opt = torch.optim.Adam(classifier.parameters())

    for i in range(epochs):
        for j, (imgs, ground_truths) in enumerate(data_loader):
            
            #Extract class label from ground truth label
            lbls = []
            for k in range(len(ground_truths)):
                lbl = ground_truths[k]
                classes, mask = torch.unique(lbl, return_inverse=True)
                classes *= 255
                max = 0
                
                #Choose the class that has the largest area
                for l in range(len(classes)):
                    if classes[l] != 0 and classes[l] != 255 and torch.sum(mask == l) > max:
                        max = l
                        
                lbls.append(classes[max])
               
            imgs = imgs.cuda()
            lbls = torch.from_numpy(np.array(lbls)).long().cuda() - 1

            #Obtain the Guided Backprop and GradCAM
            gc = grad_cam(classifier, imgs)
            gbp = guided_bp(classifier, imgs)
            
            #Input to the segmenter is the original image + gc + gbp (5 channels)
            seg_input = torch.cat((imgs, gc, gbp), 1).cuda()
            
            #Invert the output mask to leave only the background
            seg_output = segmenter(seg_input)[:,lbls[0],:,:]
            masks = (seg_output < 0.9).type(torch.cuda.FloatTensor)
            
            seg_imgs = []
            
            for k in range(imgs.shape[0]):
                seg_imgs.append(imgs[k] * masks[k,k])
            
            #Segmentor tries to increase the classifier loss after extracting the target class
            seg_imgs = torch.stack(seg_imgs)
            s_loss = -loss(classifier(seg_imgs), lbls)
            
            gen_opt.zero_grad()
            s_loss.backward()
            gen_opt.step()

            #Classifier tries to decrease loss on both original images and segmented images
            c_loss = loss(classifier(imgs), lbls) + loss(classifier(seg_imgs), lbls)
            
            clas_opt.zero_grad()
            c_loss.backward()
            clas_opt.step()
            
            if j % 20 == 0:
                print("Epoch {} Iter {} - Segmentation Loss: {}     Classificaton Loss: {}".format(i, j, s_loss*-1, c_loss))
                
    return classifier, segmenter