import torch
import dataloader from utils
import * from visualizations

def train(classifier, segmenter, epochs=10):
    data_loader = dataloader()
    loss = torch.nn.CrossEntropyLoss()
    gen_opt = torch.optim.Adam(segmenter.parameters())
    clas_opt = torch.optim.Adam(classifier.parameters())

    for i in range(epochs):
        for j, (imgs, lbls) in enumerate(data_loader):
            
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            
            gc = grad_cam(classifier, imgs)
            gbp = guided_bp(classifier, imgs)
            
            seg_input = torch.cat((imgs, gc, gbp), 1).cuda()
            
            seg_output = segmenter(seg_input)[:,lbls,:,:]
            masks = (seg_output < 0.9).type(torch.cuda.FloatTensor)
            
            seg_imgs = []
            
            for k in range(imgs.shape[0]):
                seg_imgs.append(imgs[k] * masks[k,k])
            
            seg_imgs = torch.stack(seg_imgs)
            s_loss = -loss(classifier(seg_imgs), lbls)
            
            gen_opt.zero_grad()
            s_loss.backward()
            gen_opt.step()
            
            c_loss = loss(classifier(imgs), lbls) + loss(classifier(seg_imgs), lbls)
            
            clas_opt.zero_grad()
            c_loss.backward()
            clas_opt.step()
            
            if j % 10 == 0:
                print("Epoch {} Iter {} - Segmentation Loss: {}     Classificaton Loss: {}".format(i, j, s_loss*-1, c_loss))
            
    return classifier, segmenter