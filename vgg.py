import torchvision.models as models
from torch import autograd, nn

class GuidedReLU(autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        output = input.clamp(min=0)
        return output

    def backward(self, grad_output):
        input = self.saved_tensors[0]
        guided_grad = grad_output.clone()
        
        guided_grad[guided_grad < 0] = 0
        guided_grad[input<0] = 0
        return guided_grad
    
class Vgg16(nn.Module):
    def __init__(self, pretrained=False):
        super(Vgg16, self).__init__()
        
        model = models.vgg16(pretrained=pretrained)
        layers = list(model.features.children())
                
        self.gradient = []
        self.last_conv = None
        
        self.features = nn.ModuleList(layers)
        self.classifier = model.classifier
        
    def forward(self, x, guided=False):
        for i, l in enumerate(self.features):
            if guided and l.__class__.__name__ == 'ReLU':
                l = GuidedReLU()
            
            x = l(x)
            
            #store the gradient and output for the last convolutional layer
            if i == 28:
                x.register_hook(lambda grad: self.gradient.append(grad))
                self.last_conv = x
                
        x = self.classifier(x.view(x.size(0),-1))
        
        return x